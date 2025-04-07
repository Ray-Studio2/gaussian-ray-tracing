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

GaussianTracer::GaussianTracer(const std::string& filename)
    : m_gsData(filename)
{
    m_context                = nullptr;
    gaussian_handle          = 0;
	gaussian_instances       = {};
    ptx_module               = 0;
    pipeline_compile_options = {};
    raygen_prog_group        = 0;
    miss_prog_group          = 0;
    hit_prog_group           = 0;
    pipeline                 = 0;
    sbt                      = {};
    stream                   = 0;

    params = {};
    params.output_buffer = nullptr;

    d_params = 0;

    particle_count = m_gsData.getVertexCount();
    alpha_min    = 0.2f;

    Icosahedron icosahedron = Icosahedron();
    vertices   = icosahedron.getVertices();
    indices    = icosahedron.getIndices();
    d_vertices = 0;
    d_indices  = 0;

    params.has_reflection_objects = false;

    current_lookat = make_float3(0.0f);
}

GaussianTracer::~GaussianTracer()
{
    if (d_vertices) CUDA_CHECK(cudaFree((void*)d_vertices));
    if (d_indices) CUDA_CHECK(cudaFree((void*)d_indices));

    if (params.d_particles) CUDA_CHECK(cudaFree((void*)params.d_particles));
    if (params.d_meshes) CUDA_CHECK(cudaFree((void*)params.d_meshes));
    if (d_params) CUDA_CHECK(cudaFree((void*)d_params));

    for (auto& mesh : meshes) {
        if (mesh.vertices) CUDA_CHECK(cudaFree(mesh.vertices));
        if (mesh.faces) CUDA_CHECK(cudaFree(mesh.faces));
    }

    if (pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    if (raygen_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    if (miss_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    if (hit_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(hit_prog_group));
    if (ptx_module) OPTIX_CHECK(optixModuleDestroy(ptx_module));

    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));

    if (m_context) OPTIX_CHECK(optixDeviceContextDestroy(m_context));

    if (primitives) delete primitives;
}

void GaussianTracer::setSize(unsigned int width, unsigned int height)
{
	params.width  = width;
	params.height = height;
}

void GaussianTracer::initializeOptix()
{
	createContext();
	createGaussiansASV1();
    buildAccelationStructure(gaussian_instances, gaussian_handle);
	createModule();
	createProgramGroups();
	createPipeline();
	createSBT();
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

void GaussianTracer::createGaussiansASV1()
{
    const size_t vertices_size_in_bytes = vertices.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        vertices.data(),
        vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    const size_t indices_size_in_bytes = indices.size() * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(),
        indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

	OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;

    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangle_input.triangleArray.numIndexTriplets = (unsigned int)indices.size() / 3;
    triangle_input.triangleArray.indexBuffer = d_indices;

    unsigned int triangleInputFlags[1] = { };
    triangle_input.triangleArray.flags = triangleInputFlags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &accel_options,
        &triangle_input,
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
        &triangle_input,
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

    for (int i = 0; i < particle_count; i++)
    {
        OptixInstance instance = {};

        GaussianIndice gsIndex;

        float x = m_gsData.m_particles[i].position.x;
        float y = m_gsData.m_particles[i].position.y;
        float z = m_gsData.m_particles[i].position.z;

        float opacity = m_gsData.m_particles[i].opacity;
        if (opacity > alpha_min) {
            gsIndex.index = i;
            m_gsIndice.push_back(gsIndex);
        }
        else
            continue;

        //float s = std::sqrt(2.0f * std::log(opacity / alpha_min));
        //float scale_0 = m_gsData.m_particles[i].scale.x;
        //float scale_1 = m_gsData.m_particles[i].scale.y;
        //float scale_2 = m_gsData.m_particles[i].scale.z;
        //float3 scale = make_float3(scale_0 * s, scale_1 * s, scale_2 * s);
        //glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale.x, scale.y, scale.z));

        //float qw = m_gsData.m_particles[i].rotation.x;
        //float qx = m_gsData.m_particles[i].rotation.y;
        //float qy = m_gsData.m_particles[i].rotation.z;
        //float qz = m_gsData.m_particles[i].rotation.w;
        //glm::quat rot_quat = glm::quat(qw, qx, qy, qz);
        //glm::mat4 rotation_matrix = glm::mat4_cast(rot_quat);

        //glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));

        //glm::mat4 _transform = translation_matrix * (rotation_matrix * scale_matrix);

		glm::mat4 transform = m_gsData.m_particles[i].transform;

        float instance_transform[12] = {
            transform[0][0], transform[1][0], transform[2][0], transform[3][0],
            transform[0][1], transform[1][1], transform[2][1], transform[3][1],
            transform[0][2], transform[1][2], transform[2][2], transform[3][2],
        };

        memcpy(instance.transform, instance_transform, sizeof(float) * 12);
        instance.instanceId        = i;
        instance.visibilityMask    = 255;
        instance.sbtOffset         = 0;
        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = gas;

        gaussian_instances.push_back(instance);
    }
}

void GaussianTracer::createGaussiansASV2()
{
    filterGaussians();

    size_t new_vertex_count = m_gsIndice.size();

    const size_t indices_size_in_bytes = indices.size() * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(),
        indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    std::vector<OptixBuildInput> triangle_inputs(new_vertex_count);
    std::vector<CUdeviceptr> d_transformed_vertices(new_vertex_count);
    std::vector<CUdeviceptr> d_transformed_indices(new_vertex_count);
    std::vector<uint32_t> triangle_input_flags(new_vertex_count);

    for (int i = 0; i < new_vertex_count; i++)
    {
        size_t idx = m_gsIndice[i].index;

        float x = m_gsData.m_particles[idx].position.x;
        float y = m_gsData.m_particles[idx].position.y;
        float z = m_gsData.m_particles[idx].position.z;

        float opacity = m_gsData.m_particles[idx].opacity;

        float s = std::sqrt(2.0f * std::log(opacity / alpha_min));
        float scale_0 = m_gsData.m_particles[idx].scale.x;
        float scale_1 = m_gsData.m_particles[idx].scale.y;
        float scale_2 = m_gsData.m_particles[idx].scale.z;
        float3 scale = make_float3(scale_0 * s, scale_1 * s, scale_2 * s);
        glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale.x, scale.y, scale.z));

        float qw = m_gsData.m_particles[idx].rotation.x;
        float qx = m_gsData.m_particles[idx].rotation.y;
        float qy = m_gsData.m_particles[idx].rotation.z;
        float qz = m_gsData.m_particles[idx].rotation.w;
        glm::quat rot_quat = glm::quat(qw, qx, qy, qz);
        glm::mat4 rotation_matrix = glm::mat4_cast(rot_quat);

        glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));

        glm::mat4 transform = translation_matrix * (rotation_matrix * scale_matrix);

        std::vector<float3> transformed_vertices;
        for (int j = 0; j < vertices.size(); j++) {
            glm::vec3 glm_vertex = glm::vec3(vertices[j].x, vertices[j].y, vertices[j].z);
            glm::vec4 glm_transformed_vertex = transform * glm::vec4(glm_vertex, 1.0f);

            transformed_vertices.push_back(
                make_float3(
                    glm_transformed_vertex.x,
                    glm_transformed_vertex.y,
                    glm_transformed_vertex.z
                )
            );
        }

        CUdeviceptr d_t_vertices;
        const size_t vertices_size_in_bytes = transformed_vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_t_vertices), vertices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_t_vertices),
            transformed_vertices.data(),
            vertices_size_in_bytes,
            cudaMemcpyHostToDevice
        ));

        d_transformed_vertices[i] = d_t_vertices;
        d_transformed_indices[i] = d_indices;

        triangle_inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_inputs[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_inputs[i].triangleArray.numVertices = static_cast<uint32_t>(transformed_vertices.size());
        triangle_inputs[i].triangleArray.vertexBuffers = &d_transformed_vertices[i];

        triangle_inputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_inputs[i].triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
        triangle_inputs[i].triangleArray.numIndexTriplets = (unsigned int)indices.size() / 3;
        triangle_inputs[i].triangleArray.indexBuffer = d_transformed_indices[i];

        triangle_input_flags[i] = 0;
        triangle_inputs[i].triangleArray.flags = &triangle_input_flags[i];
        triangle_inputs[i].triangleArray.numSbtRecords = 1;
        triangle_inputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
        triangle_inputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangle_inputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &accel_options,
        triangle_inputs.data(),
        triangle_inputs.size(),
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
        triangle_inputs.data(),
        triangle_inputs.size(),
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

    OptixInstance instance = {};

    float instance_transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    memcpy(instance.transform, instance_transform, sizeof(float) * 12);
    instance.instanceId = 0;
    instance.visibilityMask = 255;
    instance.sbtOffset = 0;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = gas;

    gaussian_instances.push_back(instance);
}

void GaussianTracer::buildAccelationStructure(std::vector<OptixInstance>& instances, OptixTraversableHandle& handle)
{
    CUdeviceptr d_instances;
    const size_t instances_size_in_bytes = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc((void**)&d_instances, instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy((void*)d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions instance_accel_options = {};
    instance_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    instance_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

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

void GaussianTracer::createModule()
{
    OptixModuleCompileOptions module_compile_options = {};

    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
#if (OPTIX_VERSION < 80000)
    // OPTIX_EXCEPTION_FLAG_DEBUG Removed in OptiX SDK 8.0.0.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#endif

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	std::vector<char> ptx_code = readData("gaussian-tracing_generated_shader.cu.ptx");

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
    sbt.missRecordCount             = RAY_TYPE_COUNT;
	sbt.hitgroupRecordBase          = d_hit_record;
	sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hit_record_size);
	sbt.hitgroupRecordCount         = 1;
}

void GaussianTracer::initParams()
{
	params.output_buffer             = nullptr;
	params.handle                    = gaussian_handle;
    params.k                         = MAX_K;
    params.t_min                     = 1e-3f;
    params.t_max                     = 1e5f;
    params.T_min                     = 0.03f;
    params.alpha_min                 = alpha_min;
    params.sh_degree_max             = 0;
	params.reflection_handle         = reflection_handle;
    params.reflection_render_normals = false;
    params.mode_fisheye              = false;

    {
        GaussianParticle* particles = new GaussianParticle[particle_count];
        for (int i = 0; i < particle_count; i++)
        {
            particles[i] = m_gsData.m_particles[i];
        }

        CUdeviceptr d_particles;
        const size_t particles_size = sizeof(GaussianParticle) * particle_count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_particles), particles_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_particles),
            particles,
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

void GaussianTracer::filterGaussians()
{
    for (int i = 0; i < particle_count; i++)
    {
        GaussianIndice gsIndex;

        float opacity = m_gsData.m_particles[i].opacity;

        if (opacity > alpha_min) {
            gsIndex.index = i;
            m_gsIndice.push_back(gsIndex);
        }
        else
            continue;
    }
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

//void GaussianTracer::buildReflectionAccelationStructure()
//{
//    CUdeviceptr d_instances;
//    const size_t instances_size_in_bytes = reflection_instances.size() * sizeof(OptixInstance);
//    CUDA_CHECK(cudaMalloc((void**)&d_instances, instances_size_in_bytes));
//    CUDA_CHECK(cudaMemcpy((void*)d_instances, reflection_instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));
//
//    OptixBuildInput instance_input = {};
//    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
//    instance_input.instanceArray.instances = d_instances;
//    instance_input.instanceArray.numInstances = static_cast<uint32_t>(reflection_instances.size());
//
//    OptixAccelBuildOptions instance_accel_options = {};
//    instance_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
//    instance_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
//
//    OptixAccelBufferSizes instance_buffer_sizes;
//    OPTIX_CHECK(optixAccelComputeMemoryUsage(
//        m_context,
//        &instance_accel_options,
//        &instance_input,
//        1,
//        &instance_buffer_sizes
//    ));
//
//    OptixTraversableHandle ias;
//    CUDA_CHECK(cudaMalloc((void**)&ias, instance_buffer_sizes.outputSizeInBytes));
//
//    CUdeviceptr d_instance_temp_buffer;
//    CUDA_CHECK(cudaMalloc((void**)&d_instance_temp_buffer, instance_buffer_sizes.tempSizeInBytes));
//
//    reflection_handle = 0;
//    OPTIX_CHECK(optixAccelBuild(
//        m_context,
//        0,
//        &instance_accel_options,
//        &instance_input,
//        1,
//        d_instance_temp_buffer,
//        instance_buffer_sizes.tempSizeInBytes,
//        ias,
//        instance_buffer_sizes.outputSizeInBytes,
//        &reflection_handle,
//        0,
//        0
//    ));
//
//    CUDA_CHECK(cudaStreamSynchronize(0));
//    CUDA_CHECK(cudaFree((void*)d_instance_temp_buffer));
//    CUDA_CHECK(cudaFree((void*)d_instances));
//}

OptixTraversableHandle GaussianTracer::createGAS(std::vector<float3> const& vs, std::vector<unsigned int> const& is)
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
    input.triangleArray.flags         = triangleInputFlags;
    input.triangleArray.numSbtRecords = 1;

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

    return gas;
}

OptixInstance GaussianTracer::createIAS(OptixTraversableHandle const& gas, glm::mat4 transform)
{
    float instance_transform[12] = {
        transform[0][0], transform[1][0], transform[2][0], transform[3][0],
        transform[0][1], transform[1][1], transform[2][1], transform[3][1],
        transform[0][2], transform[1][2], transform[2][2], transform[3][2]
    };

	OptixInstance instance = {};
	memcpy(instance.transform, instance_transform, sizeof(float) * 12);
	instance.instanceId        = primitives->getMeshCount() - 1;
    instance.visibilityMask    = 255;
    instance.sbtOffset         = 0;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = gas;

	return instance;
}

void GaussianTracer::removePrimitive()
{
    reflection_instances.clear();
    for (auto& mesh : meshes) {
        if (mesh.vertices) CUDA_CHECK(cudaFree(mesh.vertices));
        if (mesh.faces) CUDA_CHECK(cudaFree(mesh.faces));
    }
    meshes.clear();

    if (params.d_meshes) {
        CUDA_CHECK(cudaFree(params.d_meshes));
        params.d_meshes = nullptr;
    }

    primitives->clearPrimitives();
    reflection_handle = 0;
    updateParamsTraversableHandle();
    params.has_reflection_objects = false;
}

void GaussianTracer::updateParamsTraversableHandle()
{
	params.reflection_handle = reflection_handle;
}

void GaussianTracer::setReflectionMeshRenderNormal(bool val)
{
    params.reflection_render_normals = val;
}

void GaussianTracer::createPlane()
{
    float3 cameraPosition = params.eye;
    float cameraWeight = 0.75f;
    float gaussianWeight = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createPlane(primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform);

	//p.gas = gas;

    reflection_instances.push_back(ias);

	buildAccelationStructure(reflection_instances, reflection_handle);
    updateParamsTraversableHandle();   

	sendGeometryAttributesToDevice(p);
    
    params.has_reflection_objects = true;
}

void GaussianTracer::createSphere()
{
    float3 cameraPosition = params.eye;
    float cameraWeight = 0.75f;
    float gaussianWeight = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createSphere(primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform);

    reflection_instances.push_back(ias);

    buildAccelationStructure(reflection_instances, reflection_handle);
    updateParamsTraversableHandle();

    sendGeometryAttributesToDevice(p);

	params.has_reflection_objects = true;
}

void GaussianTracer::createLoadMesh(std::string filename)
{
    float3 cameraPosition = params.eye;
    float cameraWeight = 0.75f;
    float gaussianWeight = 1.0f - cameraWeight;

    float3 primitive_position = {
        current_lookat.x * gaussianWeight + cameraPosition.x * cameraWeight,
        current_lookat.y * gaussianWeight + cameraPosition.y * cameraWeight,
        current_lookat.z * gaussianWeight + cameraPosition.z * cameraWeight
    };

    Primitive p = primitives->createLoadMesh(filename, primitive_position);

    OptixTraversableHandle gas = createGAS(p.vertices, p.indices);
    OptixInstance          ias = createIAS(gas, p.transform);

    reflection_instances.push_back(ias);

    buildAccelationStructure(reflection_instances, reflection_handle);
    updateParamsTraversableHandle();

    sendGeometryAttributesToDevice(p);

    params.has_reflection_objects = true;
}

void GaussianTracer::sendGeometryAttributesToDevice(Primitive p)
{
    size_t vertex_count = p.vertex_count;
    Vertex* _vertices = new Vertex[vertex_count];
    for (int i = 0; i < vertex_count; i++) {
        Vertex v;
        glm::vec3 glm_position = glm::vec3(p.vertices[i].x, p.vertices[i].y, p.vertices[i].z);
        glm::vec4 glm_transformed_position = p.transform * glm::vec4(glm_position, 1.0f);

        glm::vec3 glm_normal = glm::vec3(p.normals[i].x, p.normals[i].y, p.normals[i].z);
        glm::vec3 glm_transformed_normal = glm::mat3(p.transform) * glm_normal;

        v.position = make_float3(glm_transformed_position.x, glm_transformed_position.y, glm_transformed_position.z);
        v.normal = make_float3(glm_transformed_normal.x, glm_transformed_normal.y, glm_transformed_normal.z);

        _vertices[i] = v;
    }

    CUdeviceptr _d_vertices;
    const size_t vertices_size = sizeof(Vertex) * vertex_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_d_vertices),
        _vertices,
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_face_indices;
    const size_t indices_size = (p.indices.size() / 3) * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_indices), indices_size));
    std::vector<uint3> face_indices;
    for (size_t i = 0; i < p.indices.size(); i += 3) {
        face_indices.push_back(make_uint3(
            p.indices[i],
            p.indices[i + 1],
            p.indices[i + 2]
        ));
    }
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_face_indices),
        face_indices.data(),
        indices_size,
        cudaMemcpyHostToDevice
    ));

	Mesh mesh;
	mesh.vertices = reinterpret_cast<Vertex*>(_d_vertices);
	mesh.faces = reinterpret_cast<Face*>(d_face_indices);

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

	reflection_instances[p.instanceIndex] = instance;

	//buildReflectionAccelationStructure();
    buildAccelationStructure(reflection_instances, reflection_handle);
    updateParamsTraversableHandle();
    updateGeometryAttributesToDevice(p);
}

void GaussianTracer::updateGeometryAttributesToDevice(Primitive& p)
{
    size_t vertex_count = p.vertex_count;
    Vertex* _vertices = new Vertex[vertex_count];
    for (int i = 0; i < vertex_count; i++) {
        Vertex v;
		glm::vec3 glm_position = glm::vec3(p.vertices[i].x, p.vertices[i].y, p.vertices[i].z);
		glm::vec4 glm_transformed_position = p.transform * glm::vec4(glm_position, 1.0f);

		glm::vec3 glm_normal = glm::vec3(p.normals[i].x, p.normals[i].y, p.normals[i].z);
		glm::vec3 glm_transformed_normal = glm::mat3(p.transform) * glm_normal;

		v.position = make_float3(glm_transformed_position.x, glm_transformed_position.y, glm_transformed_position.z);
		v.normal = make_float3(glm_transformed_normal.x, glm_transformed_normal.y, glm_transformed_normal.z);

        _vertices[i] = v;
    }

    CUdeviceptr _d_vertices;
    const size_t vertices_size = sizeof(Vertex) * vertex_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(_d_vertices),
        _vertices,
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_face_indices;
    const size_t indices_size = (p.indices.size() / 3) * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_indices), indices_size));
    std::vector<uint3> face_indices;
    for (size_t i = 0; i < p.indices.size(); i += 3) {
        face_indices.push_back(make_uint3(
            p.indices[i],
            p.indices[i + 1],
            p.indices[i + 2]
        ));
    }
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_face_indices),
        face_indices.data(),
        indices_size,
        cudaMemcpyHostToDevice
    ));

    Mesh mesh;
    mesh.vertices = reinterpret_cast<Vertex*>(_d_vertices);
    mesh.faces = reinterpret_cast<Face*>(d_face_indices);

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