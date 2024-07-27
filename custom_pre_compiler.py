import textwrap

side = 500
threadsPerBlock = 256

def generateListOfCases():
    cases = []
    # Core
    cases.append(('CORE', [False] * 6))
    # Faces
    faces = []
    i = 0
    for direction in ['X', 'Y', 'Z']:
        for face in ['1', '2']:
            boundaryFlags = [False] * 6
            boundaryFlags[i] = True
            faces.append((direction + face, boundaryFlags))
            cases.append((direction + face, boundaryFlags))
            i += 1
    # Edges are intersections of faces
    for face1 in faces:
        for face2 in faces:
            if face[0][0] != face[0][0]:
                cases.append((face1[0] + face2[0], face1[1] + face2[1]))
    # Corners
    for face1 in faces:
        for face2 in faces:
            for face3 in faces:
                if face1[0][0] != face2[0][0] and face2[0][0] != face3[0][0] and face1[0][0] != face3[0][0]:
                    cases.append((face1[0] + face2[0] + face3[0], face1[1] + face2[1] + face3[1]))
    return cases

def inlineSingleDerivative(field, deriv, boundaryFlags):
    if deriv == '0':
        if boundaryFlags[0]:
            return f'({field}[index_xp] - {field}[index]) / dx'
        elif boundaryFlags[1]:
            return f'({field}[index] - {field}[index_xm]) / dx'
        else:
            return f'({field}[index_xp] - {field}[index_xm]) / (2 * dx)'
    elif deriv == '1':
        if boundaryFlags[2]:
            return f'({field}[index_yp] - {field}[index]) / dx'
        elif boundaryFlags[3]:
            return f'({field}[index] - {field}[index_ym]) / dx'
        else:
            return f'({field}[index_yp] - {field}[index_ym]) / (2 * dx)'
    elif deriv == '2':
        if boundaryFlags[4]:
            return f'({field}[index_zp] - {field}[index]) / dx'
        elif boundaryFlags[5]:
            return f'({field}[index] - {field}[index_zm]) / dx'
        else:
            return f'({field}[index_zp] - {field}[index_zm]) / (2 * dx)'
    elif deriv == '3':
        if boundaryFlags[0]:
            return f'({field}[index_xpp] - 2 * {field}[index_xp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[1]:
            return f'({field}[index] - 2 * {field}[index_xm] + {field}[index_xmm]) / (dx * dx)'
        else:
            return f'({field}[index_xp] - 2 * {field}[index] + {field}[index_xm]) / (dx * dx)'
    elif deriv == '4':
        if boundaryFlags[2]:
            return f'({field}[index_ypp] - 2 * {field}[index_yp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[3]:
            return f'({field}[index] - 2 * {field}[index_ym] + {field}[index_ymm]) / (dx * dx)'
        else:
            return f'({field}[index_yp] - 2 * {field}[index] + {field}[index_ym]) / (dx * dx)'
    elif deriv == '5':
        if boundaryFlags[4]:
            return f'({field}[index_zpp] - 2 * {field}[index_zp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[5]:
            return f'({field}[index] - 2 * {field}[index_zm] + {field}[index_zmm]) / (dx * dx)'
        else:
            return f'({field}[index_zp] - 2 * {field}[index] + {field}[index_zm]) / (dx * dx)'

def inlineDerivatives(kernel, boundaryFlags):
    newKernel = ''
    lines = kernel.split('\n')
    for line in lines:
        if 'INLINE_DERIV' in line:
            lineBefore = line.split('INLINE_DERIV')[0]
            lineAfter = line.split('INLINE_DERIV')[1].split(')')[1]
            funcArgs = line.split('INLINE_DERIV')[1]
            funcArgs = funcArgs.split('(')[1].split(')')[0].split(',')
            field = funcArgs[0].strip()
            deriv = funcArgs[1].strip()
            newKernel += lineBefore + inlineSingleDerivative(field, deriv, boundaryFlags) + lineAfter + '\n'
        else:
            newKernel += line + '\n'
    return newKernel

def inlineKernelBody(kernel):
    kernel_body = f"""
    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * INLINE_DERIV(f1.u, 0)
        - f1.v[index] * INLINE_DERIV(f1.u, 1)
        - f1.w[index] * INLINE_DERIV(f1.u, 2)
        - (RT/f1.p[index]) * INLINE_DERIV(f1.p, 0)
        + mu * (RT/f1.p[index]) * (
            INLINE_DERIV(f1.u, 3) +
            INLINE_DERIV(f1.u, 4) +
            INLINE_DERIV(f1.u, 5)
        )
    );
    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * INLINE_DERIV(f1.v, 0)
        - f1.v[index] * INLINE_DERIV(f1.v, 1)
        - f1.w[index] * INLINE_DERIV(f1.v, 2)
        - (RT/f1.p[index]) * INLINE_DERIV(f1.p, 1)
        + mu * (RT/f1.p[index]) * (
            INLINE_DERIV(f1.v, 3) +
            INLINE_DERIV(f1.v, 4) +
            INLINE_DERIV(f1.v, 5)
        )
    );
    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * INLINE_DERIV(f1.w, 0)
        - f1.v[index] * INLINE_DERIV(f1.w, 1)
        - f1.w[index] * INLINE_DERIV(f1.w, 2)
        - (RT/f1.p[index]) * INLINE_DERIV(f1.p, 2)
        + mu * (RT/f1.p[index]) * (
            INLINE_DERIV(f1.w, 3) +
            INLINE_DERIV(f1.w, 4) +
            INLINE_DERIV(f1.w, 5)
        )
    );
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
        (INLINE_DERIV(f1.u, 0) * f1.p[index] +
        INLINE_DERIV(f1.p, 0) * f1.u[index] +
        INLINE_DERIV(f1.v, 1) * f1.p[index] +
        INLINE_DERIV(f1.p, 1) * f1.v[index] +
        INLINE_DERIV(f1.w, 2) * f1.p[index] +
        INLINE_DERIV(f1.p, 2) * f1.w[index]);
    """
    kernel_body = textwrap.dedent(kernel_body)    
    kernel_body = textwrap.indent(kernel_body, ' ' * 8)
    return kernel.replace('INLINE_KERNEL_BODY', kernel_body)

def generateCoreKernel(boundaryFlags):
    coreSide = side - 2
    coreSize = coreSide ** 3
    kernel = ''
    kernel += f"""
    __global__ void updateCore(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= {coreSize}) return;

        // Compute indices
        int x = (totalThreadIndex % {coreSide}) + 1;
        int y = ((totalThreadIndex / {coreSide}) % {coreSide}) + 1;
        int z = (totalThreadIndex / ({coreSide} * {coreSide})) + 1;
        int index = x + y * {side} + z * {side} * {side};
        int index_xm = (x-1) + y * {side} + z * {side} * {side};
        int index_xp = (x+1) + y * {side} + z * {side} * {side};
        int index_ym = x + (y-1) * {side} + z * {side} * {side};
        int index_yp = x + (y+1) * {side} + z * {side} * {side};
        int index_zm = x + y * {side} + (z-1) * {side} * {side};
        int index_zp = x + y * {side} + (z+1) * {side} * {side};

        INLINE_KERNEL_BODY
    }}
    """
    kernel = inlineKernelBody(kernel)
    kernel = inlineDerivatives(kernel, boundaryFlags)
    return kernel

def inlineFaceIndexing(kernel, face, faceSide):
    if face[0] == 'X':
        indexing = f"""
        int x = 0;
        int y = (totalThreadIndex % {faceSide}) + 1;
        int z = ((totalThreadIndex / {faceSide}) % {faceSide}) + 1;
        int index = x + y * {side} + z * {side} * {side};
        int index_xm = x + y * {side} + z * {side} * {side};
        int index_xp = (side - 1) + y * {side} + z * {side} * {side};
        int index_ym = x + (y-1) * {side} + z * {side} * {side};
        int index_yp = x + (y+1) * {side} + z * {side} * {side};
        int index_zm = x + y * {side} + (z-1) * {side} * {side};
        int index_zp = x + y * {side} + (z+1) * {side} * {side};
        """

def generateFaceKernel(face, boundaryFlags):
    faceSide = side - 2
    faceSize = faceSide ** 2
    kernel = ''
    kernel += f"""
    __global__ void updateFace{face}(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= {faceSize}) return;

        INLINE_FACE_INDEXING

        INLINE_KERNEL_BODY
    }}
    """
    kernel = inlineKernelBody(kernel)
    kernel = inlineDerivatives(kernel, boundaryFlags)
    kernel = inlineFaceIndexing(kernel, face, faceSide)
    print(kernel)
    return kernel

def generateEdgeKernel(edge):
    edgeSize = side - 2
    kernel = ''
    kernel += f"""
    __global__ void updateCore(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= {edgeSize}) return;

        INLINE_EDGE_INDEXING

        INLINE_KERNEL_BODY
    }}
    """
    return kernel

def generateCornerKernel(corner):
    kernel = ''
    kernel += f"""
    __global__ void updateCore(data d) {{
        INLINE_FACE_INDEXING

        INLINE_KERNEL_BODY
    }}
    """
    return kernel

def generateListofKernels(cases):
    kernels = []
    for case in cases:
        if case[0] == 'CORE':
            kernels.append(generateCoreKernel(case[1]))
        elif len(case[0]) == 2:
            # Face
            kernels.append(generateFaceKernel(*case))
        '''elif len(case[0]) == 4:
            # Edge
            kernels.append(generateEdgeKernel(*case))
        elif len(case[0]) == 6:
            # Corner
            kernels.append(generateCornerKernel(*case))
        else:
            raise ValueError('Invalid case length')'''

def preCompile():
    # This function is called before building the CUDA/C++ code
    # It is used to automatically generate high-performance but unmaintainable code to handle CFD updates in the cube's core,
    # faces, edges, and corners.
    cases = generateListOfCases()
    kernels = generateListofKernels(cases)

preCompile()