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

GaussianTracer::GaussianTracer(const std::string& filename)
	: m_gsData(filename)
{
	m_context      = nullptr;
    triangle_input = {};
	m_gas          = 0;

    ptx_module               = 0;
	pipeline_compile_options = {};
	raygen_prog_group        = 0;
	miss_prog_group          = 0;
	anyhit_prog_group        = 0;
	pipeline                 = 0;
	sbt                      = {};
	stream                   = 0;

    params = {};
	params.output_buffer = nullptr;

    d_params = 0;

	vertex_count = m_gsData.getVertexCount();
	alpha_min    = 0.2f;

    Icosahedron icosahedron = Icosahedron();
	vertices   = icosahedron.getVertices();
	indices    = icosahedron.getIndices();
    d_vertices = 0;
	d_indices  = 0;
}

GaussianTracer::~GaussianTracer()
{

}

void GaussianTracer::setSize(unsigned int width, unsigned int height)
{
	params.width  = width;
	params.height = height;
}

void GaussianTracer::initializeOptix()
{
	createContext();
    buildAccelationStructure();
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

void GaussianTracer::buildAccelationStructure()
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
		d_transformed_indices[i]  = d_indices;

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
        &m_gas,
        0,
        0
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer));
}

void GaussianTracer::createModule()
{
    OptixModuleCompileOptions module_compile_options = {};

    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	std::vector<char> ptx_code = readData("../gaussain-tracing_generated_shader.cu.ptx");

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
        OptixProgramGroupDesc anyhit_prog_group_desc = {};

        anyhit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        anyhit_prog_group_desc.hitgroup.moduleAH = ptx_module;
        anyhit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__anyhit";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &anyhit_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &anyhit_prog_group
        ));
    }
}

void GaussianTracer::createPipeline()
{
    OptixProgramGroup program_groups[] =
    {
        raygen_prog_group,
        anyhit_prog_group,
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
    OPTIX_CHECK(optixUtilAccumulateStackSizes(anyhit_prog_group, &stack_sizes, pipeline));
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

    const uint32_t max_traversal_depth = 1;
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

    std::vector<AnyHitRecord> anyhit_records;
    for (int i = 0; i < m_gsIndice.size(); i++)
    {
        AnyHitRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(anyhit_prog_group, &rec));
		rec.data.index = m_gsIndice[i].index;
		anyhit_records.push_back(rec);
    }

    CUdeviceptr d_anyhit_records;
    const size_t anyhit_record_size = sizeof(AnyHitRecord) * anyhit_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_anyhit_records), anyhit_record_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_anyhit_records),
        anyhit_records.data(),
        anyhit_record_size,
        cudaMemcpyHostToDevice
    ));

    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = d_anyhit_records;
    sbt.hitgroupRecordStrideInBytes = sizeof(AnyHitRecord);
    sbt.hitgroupRecordCount = static_cast<uint32_t>(anyhit_records.size());
}

void GaussianTracer::initParams()
{
	params.output_buffer = nullptr;
    params.handle        = m_gas;
    params.k             = MAX_K;
    params.t_min         = 1e-3f;
    params.t_max         = 1e5f;
    params.T_min         = 0.03f;
    params.alpha_min     = alpha_min;
    params.sh_degree_max = 0;

	GaussianParticle* particles = new GaussianParticle[vertex_count];
	for (int i = 0; i < vertex_count; i++)
	{
		particles[i] = m_gsData.m_particles[i];
	}

    CUdeviceptr d_particles;
    const size_t particles_size = sizeof(GaussianParticle) * vertex_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_particles), particles_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_particles),
        particles,
        particles_size,
        cudaMemcpyHostToDevice
    ));
    params.d_particles = reinterpret_cast<GaussianParticle*>(d_particles);

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

void GaussianTracer::render(CUDAOutputBuffer& output_buffer)
{
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
    for (int i = 0; i < vertex_count; i++)
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

void GaussianTracer::initCamera()
{
    m_camera.setEye(make_float3(0.0f, 0.0f, 3.0f));
    m_camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    m_camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    m_camera.setFovY(60.0f);
    m_camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));

	m_camera_changed = true;
}

void GaussianTracer::updateCamera()
{
	if (!m_camera_changed)
		return;

	params.eye = m_camera.eye();
	m_camera.UVWFrame(params.U, params.V, params.W);
}