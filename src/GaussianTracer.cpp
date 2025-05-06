#include "GaussianTracer.h"
#include "Exception.h"
#include "Utility.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#include <vector>
#include <random>

#include "vector_math.h"

GaussianTracer::GaussianTracer(const std::string& filename)
    : m_gsData(filename)
{
    // Optix state
    m_context                = nullptr;
    ptx_module               = 0;
    pipeline_compile_options = {};
    raygen_prog_group        = 0;
    miss_prog_group          = 0;
    hit_prog_group           = 0;
    pipeline                 = 0;
    sbt                      = {};
    
    stream = 0;

    // Geometry
    gaussian_handle    = 0;
    gaussian_instances = {};
	mesh_handle        = 0;
	mesh_instances     = {};

    params   = {};
    d_params = 0;

    particle_count = m_gsData.getVertexCount();
    alpha_min    = 0.01f;

    Icosahedron icosahedron = Icosahedron();
    vertices   = icosahedron.getVertices();
    indices    = icosahedron.getIndices();

    current_lookat = make_float3(0.0f);
}

GaussianTracer::~GaussianTracer()
{
    if (params.d_particles) CUDA_CHECK(cudaFree((void*)params.d_particles));
    if (params.d_meshes) CUDA_CHECK(cudaFree((void*)params.d_meshes));
    if (d_params) CUDA_CHECK(cudaFree((void*)d_params));

    //for (auto& mesh : meshes) {
    //    if (mesh.vertices) CUDA_CHECK(cudaFree(mesh.vertices));
    //    if (mesh.faces) CUDA_CHECK(cudaFree(mesh.faces));
    //}

    if (pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    if (raygen_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    if (miss_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    if (hit_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(hit_prog_group));
    if (ptx_module) OPTIX_CHECK(optixModuleDestroy(ptx_module));

    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));

    if (m_context) OPTIX_CHECK(optixDeviceContextDestroy(m_context));

    if (primitives) delete primitives;
}

void GaussianTracer::initializeOptix()
{
	createContext();
	createModule();
	createProgramGroups();
	createPipeline();
	createSBT();

    createGaussianParticlesBVH();
    
    initializeParams();
}

void GaussianTracer::createContext()
{
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext cuCtx = 0;

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    m_context = context;
}

void GaussianTracer::createModule()
{
    OptixModuleCompileOptions module_compile_options = {};

    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 14;
    pipeline_compile_options.numAttributeValues = 0;
#if (OPTIX_VERSION < 80000)
    // OPTIX_EXCEPTION_FLAG_DEBUG Removed in OptiX SDK 8.0.0.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	std::vector<char> ptx_code = readData("gaussian-tracing_generated_tracer.cu.ptx");

    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code.data(),
        ptx_code.size(),
        LOG,
        &LOG_SIZE,
        &ptx_module
    ));
}

void GaussianTracer::createProgramGroups()
{
    OptixProgramGroupOptions  program_group_options = {};
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__raygeneration";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__miss";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &miss_prog_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};

        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleAH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__anyhit";
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__closesthit";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &hit_prog_group
        ));
    }
}

void GaussianTracer::createPipeline()
{
    OptixProgramGroup program_groups[] =
    {
        raygen_prog_group,
        hit_prog_group,
        miss_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog_group, &stack_sizes, pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(hit_prog_group, &stack_sizes, pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(miss_prog_group, &stack_sizes, pipeline));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}

void GaussianTracer::createSBT()
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));
    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_miss_record;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));
    MissRecord ms_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_hit_record;
    const size_t hit_record_size = sizeof(HitRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), hit_record_size));
    HitRecord hit_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_prog_group, &hit_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hit_record),
        &hit_sbt,
        hit_record_size,
        cudaMemcpyHostToDevice
    ));

    sbt.raygenRecord                = d_raygen_record;
    sbt.missRecordBase              = d_miss_record;
    sbt.missRecordStrideInBytes     = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount             = 1;
	sbt.hitgroupRecordBase          = d_hit_record;
	sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hit_record_size);
	sbt.hitgroupRecordCount         = 1;
}

void GaussianTracer::createGaussianParticlesBVH()
{
    OptixTraversableHandle gas = createGAS(vertices, indices);

    for (int i = 0; i < particle_count; i++) {
        OptixInstance instance = {};

        float opacity = m_gsData.particles[i].opacity;

        float s = sqrtf(2.0f * logf(opacity / alpha_min));
        glm::mat4 scale_matrix       = glm::scale(glm::mat4(1.0f), m_gsData.particles[i].scale * s);
        glm::mat4 rotation_matrix    = glm::mat4_cast(m_gsData.particles[i].rotation);
        glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), m_gsData.particles[i].position);
        
        glm::mat4 transform = translation_matrix * (rotation_matrix * scale_matrix);

        gaussian_instances.push_back(createIAS(gas, transform, i));
    }

    buildAccelationStructure(gaussian_instances, gaussian_handle);
}

OptixTraversableHandle GaussianTracer::createGAS(std::vector<float3> const& vs,
    std::vector<unsigned int> const& is)
{
    CUdeviceptr d_vs;
    CUdeviceptr d_is;

    const size_t vertices_size_in_bytes = vs.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vs), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vs),
        vs.data(),
        vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    const size_t indices_size_in_bytes = is.size() * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_is), indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_is),
        is.data(),
        indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.numVertices         = vs.size();
    input.triangleArray.vertexBuffers       = &d_vs;

    input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    input.triangleArray.numIndexTriplets   = (unsigned int)is.size() / 3;
    input.triangleArray.indexBuffer        = d_is;

    unsigned int triangleInputFlags[1] = {};
    input.triangleArray.flags          = triangleInputFlags;
    input.triangleArray.numSbtRecords  = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &accel_options,
        &input,
        1,
        &gas_buffer_sizes
    ));

    CUdeviceptr d_gas;
    CUDA_CHECK(cudaMalloc((void**)&d_gas, gas_buffer_sizes.outputSizeInBytes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));

    OptixTraversableHandle gas;
    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0,
        &accel_options,
        &input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas,
        gas_buffer_sizes.outputSizeInBytes,
        &gas,
        0,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer));
    CUDA_CHECK(cudaFree((void*)d_vs));
    CUDA_CHECK(cudaFree((void*)d_is));
    return gas;
}

OptixInstance GaussianTracer::createIAS(OptixTraversableHandle const& gas,
    glm::mat4 transform,
    size_t index)
{
    float instance_transform[12] = {
        transform[0][0], transform[1][0], transform[2][0], transform[3][0],
        transform[0][1], transform[1][1], transform[2][1], transform[3][1],
        transform[0][2], transform[1][2], transform[2][2], transform[3][2]
    };

    OptixInstance instance = {};
    memcpy(instance.transform, instance_transform, sizeof(float) * 12);
    instance.instanceId        = index;
    instance.visibilityMask    = 255;
    instance.sbtOffset         = 0;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = gas;

    return instance;
}

void GaussianTracer::buildAccelationStructure(std::vector<OptixInstance>& instances, OptixTraversableHandle& handle)
{
    CUdeviceptr d_instances;
    const size_t instances_size_in_bytes = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc((void**)&d_instances, instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy((void*)d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions instance_accel_options = {};
    instance_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    instance_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes instance_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &instance_accel_options,
        &instance_input,
        1,
        &instance_buffer_sizes
    ));

    OptixTraversableHandle ias;

    CUDA_CHECK(cudaMalloc((void**)&ias, instance_buffer_sizes.outputSizeInBytes));

    CUdeviceptr d_instance_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_instance_temp_buffer, instance_buffer_sizes.tempSizeInBytes));

    handle = 0;
    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0,
        &instance_accel_options,
        &instance_input,
        1,
        d_instance_temp_buffer,
        instance_buffer_sizes.tempSizeInBytes,
        ias,
        instance_buffer_sizes.outputSizeInBytes,
        &handle,
        0,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)d_instance_temp_buffer));
    CUDA_CHECK(cudaFree((void*)d_instances));
}

void GaussianTracer::initializeParams()
{
    params.output_buffer             = nullptr;
    params.handle                    = gaussian_handle;
    params.k                         = MAX_K;
    params.t_min                     = 1e-3f;
    params.t_max                     = 1e5f;
    params.minTransmittance          = 0.001f;
    params.alpha_min                 = alpha_min;
    params.sh_degree_max             = 0;
    params.mesh_handle               = mesh_handle;
    params.reflection_render_normals = false;
    params.mode_fisheye              = false;

    params.type       = MIRROR;
	params.traceState = TraceLastGaussianPass;

    {
        CUdeviceptr d_particles;
        const size_t particles_size = sizeof(GaussianParticle) * particle_count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_particles), particles_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_particles),
            m_gsData.particles.data(),
            particles_size,
            cudaMemcpyHostToDevice
        ));
        params.d_particles = reinterpret_cast<GaussianParticle*>(d_particles);
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

void GaussianTracer::render(CUDAOutputBuffer& output_buffer)
{
    if (params.mode_fisheye) {
        CUDA_CHECK(cudaMemset(output_buffer.map(), 0, params.width * params.height * sizeof(uchar3)));
        output_buffer.unmap();
    }

    uchar3* result_buffer_data = output_buffer.map();
    params.output_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(d_params),
        &params, 
        sizeof(Params),
        cudaMemcpyHostToDevice, 
        stream
    ));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(Params),
        &sbt,
        params.width,   // launch width
        params.height,  // launch height
        1               // launch depth
    ));

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void GaussianTracer::updateCamera(Camera& camera, bool& camera_changed)
{
	if (!camera_changed)
		return;
    camera_changed = false;

	camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
	params.eye = camera.eye();
	camera.UVWFrame(params.U, params.V, params.W);

    current_lookat = camera.lookat();
}

void GaussianTracer::removePrimitive()
{
    mesh_instances.clear();
    //for (auto& mesh : meshes) {
    //    if (mesh.vertices) CUDA_CHECK(cudaFree(mesh.vertices));
    //    if (mesh.faces) CUDA_CHECK(cudaFree(mesh.faces));
    //}
    meshes.clear();

    if (params.d_meshes) {
        CUDA_CHECK(cudaFree(params.d_meshes));
        params.d_meshes = nullptr;
    }

    primitives->clearPrimitives();
    mesh_handle = 0;
    updateParamsTraversableHandle();
}

void GaussianTracer::updateParamsTraversableHandle()
{
	params.mesh_handle = mesh_handle;
}

void GaussianTracer::setReflectionMeshRenderNormal(bool val)
{
    params.reflection_render_normals = val;
}

void GaussianTracer::createPlane()
{
    float3 cameraPosition = params.eye;
    float cameraWeight    = 0.75f;
    float gaussianWeight  = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createPlane(primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform, primitives->getMeshCount() - 1);

    mesh_instances.push_back(ias);

	buildAccelationStructure(mesh_instances, mesh_handle);
    updateParamsTraversableHandle();   

	sendGeometryAttributesToDevice(p);
}

void GaussianTracer::createSphere()
{
    float3 cameraPosition = params.eye;
    float cameraWeight    = 0.75f;
    float gaussianWeight  = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createSphere(primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform, primitives->getMeshCount() - 1);

    mesh_instances.push_back(ias);

    buildAccelationStructure(mesh_instances, mesh_handle);
    updateParamsTraversableHandle();

    sendGeometryAttributesToDevice(p);
}

void GaussianTracer::createLoadMesh(std::string filename)
{
    float3 cameraPosition = params.eye;
    float cameraWeight    = 0.75f;
    float gaussianWeight  = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createLoadMesh(filename, primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform, primitives->getMeshCount() - 1);

    mesh_instances.push_back(ias);

    buildAccelationStructure(mesh_instances, mesh_handle);
    updateParamsTraversableHandle();

    sendGeometryAttributesToDevice(p);
}

void GaussianTracer::sendGeometryAttributesToDevice(Primitive p)
{
    size_t vertex_count = p.vertex_count;
    float3* vertex_normals = new float3[vertex_count];
    std::vector<uint3> faces;
    for (int i = 0; i < vertex_count; i++) {
        glm::vec3 glm_normal = glm::vec3(p.normals[i].x, p.normals[i].y, p.normals[i].z);
        glm::vec3 glm_transformed_normal = glm::mat3(p.transform) * glm_normal;

        vertex_normals[i] = make_float3(glm_transformed_normal.x, glm_transformed_normal.y, glm_transformed_normal.z);
    }
    for (size_t i = 0; i < p.indices.size(); i += 3) {
        faces.push_back(make_uint3(
            p.indices[i],
            p.indices[i + 1],
            p.indices[i + 2]
        ));
    }

    CUdeviceptr _d_vertices;
    const size_t vertices_size = sizeof(float3) * vertex_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_d_vertices),
        vertex_normals,
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_face_indices;
	const size_t indices_size = sizeof(uint3) * (p.indices.size() / 3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_face_indices),
        faces.data(),
        indices_size,
        cudaMemcpyHostToDevice
    ));

	Mesh mesh;
    mesh.faces = reinterpret_cast<uint3*>(d_face_indices);
	mesh.vertex_normals = reinterpret_cast<float3*>(_d_vertices);

	meshes.push_back(mesh);

	CUdeviceptr d_meshes;
	const size_t mesh_size = sizeof(Mesh) * meshes.size();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_meshes), mesh_size));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_meshes),
		meshes.data(),
        mesh_size,
		cudaMemcpyHostToDevice
	));

	params.d_meshes = reinterpret_cast<Mesh*>(d_meshes);
}

void GaussianTracer::updateInstanceTransforms(Primitive& p)
{
    glm::mat4 transform = p.transform;

    float instance_transform[12] = {
        transform[0][0], transform[1][0], transform[2][0], transform[3][0],
        transform[0][1], transform[1][1], transform[2][1], transform[3][1],
        transform[0][2], transform[1][2], transform[2][2], transform[3][2]
    };

    OptixInstance instance = {};
    memcpy(instance.transform, instance_transform, sizeof(float) * 12);
	instance.instanceId        = p.instanceIndex;
    instance.visibilityMask    = 255;
    instance.sbtOffset         = 0;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
	// TODO: Better way to get the gas handle
    instance.traversableHandle = createGAS(p.vertices, p.indices);

    mesh_instances[p.instanceIndex] = instance;

	//buildReflectionAccelationStructure();
    buildAccelationStructure(mesh_instances, mesh_handle);
    updateParamsTraversableHandle();
    updateGeometryAttributesToDevice(p);
}

void GaussianTracer::updateGeometryAttributesToDevice(Primitive& p)
{
    size_t vertex_count = p.vertex_count;
	float3* vertex_normals = new float3[vertex_count];
    std::vector<uint3> faces;
    for (int i = 0; i < vertex_count; i++) {
		glm::vec3 glm_normal = glm::vec3(p.normals[i].x, p.normals[i].y, p.normals[i].z);
		glm::vec3 glm_transformed_normal = glm::mat3(p.transform) * glm_normal;

        vertex_normals[i] = make_float3(glm_transformed_normal.x, glm_transformed_normal.y, glm_transformed_normal.z);
    }
    for (size_t i = 0; i < p.indices.size(); i += 3) {
        faces.push_back(make_uint3(
            p.indices[i],
            p.indices[i + 1],
            p.indices[i + 2]
        ));
    }

    CUdeviceptr _d_vertices;
    const size_t vertices_size = sizeof(float3) * vertex_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_d_vertices),
        vertex_normals,
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_face_indices;
    const size_t indices_size = sizeof(uint3) * (p.indices.size() / 3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_face_indices),
        faces.data(),
        indices_size,
        cudaMemcpyHostToDevice
    ));

    Mesh mesh;
    mesh.faces = reinterpret_cast<uint3*>(d_face_indices);
    mesh.vertex_normals = reinterpret_cast<float3*>(_d_vertices);

	meshes[p.instanceIndex] = mesh;

    CUdeviceptr d_meshes;
    const size_t mesh_size = sizeof(Mesh) * meshes.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_meshes), mesh_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_meshes),
        meshes.data(),
        mesh_size,
        cudaMemcpyHostToDevice
    ));

    params.d_meshes = reinterpret_cast<Mesh*>(d_meshes);
}

void GaussianTracer::setSize(unsigned int width, unsigned int height)
{
    params.width = width;
    params.height = height;
}