import textwrap

side = 300
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
    facesXY = [face for face in faces if face[0][0] in ['X', 'Y']]
    facesXZ = [face for face in faces if face[0][0] in ['X', 'Z']]
    for i in range(len(facesXY)):
        for j in range(i + 1, len(facesXY)):
            face1 = facesXY[i]
            face2 = facesXY[j]
            if face1[0][0] != face2[0][0]:
                cases.append((face1[0] + face2[0], [face1[1][i] or face2[1][i] for i in range(6)]))
    for face1 in facesXZ:
        for face2 in facesXZ:
            if face1[0][0] != face2[0][0]:
                cases.append((face1[0] + face2[0], [face1[1][i] or face2[1][i] for i in range(6)]))
    # Corners
    facesZ = [face for face in faces if face[0][0] == 'Z']
    for face1 in facesZ:
        for i in range(len(facesXY)):
            for j in range(i + 1, len(facesXY)):
                face2 = facesXY[i]
                face3 = facesXY[j]
                if face2[0][0] != face3[0][0]:
                    cases.append((face1[0] + face2[0] + face3[0], [face1[1][i] or face2[1][i] or face3[1][i] for i in range(6)]))
    return cases

def getDeriv(field, deriv, boundaryFlags):
    if deriv == 0:
        if boundaryFlags[0]:
            return f'({field}[index_xp] - {field}[index]) / dx'
        elif boundaryFlags[1]:
            return f'({field}[index] - {field}[index_xm]) / dx'
        else:
            return f'({field}[index_xp] - {field}[index_xm]) / (2 * dx)'
    elif deriv == 1:
        if boundaryFlags[2]:
            return f'({field}[index_yp] - {field}[index]) / dx'
        elif boundaryFlags[3]:
            return f'({field}[index] - {field}[index_ym]) / dx'
        else:
            return f'({field}[index_yp] - {field}[index_ym]) / (2 * dx)'
    elif deriv == 2:
        if boundaryFlags[4]:
            return f'({field}[index_zp] - {field}[index]) / dx'
        elif boundaryFlags[5]:
            return f'({field}[index] - {field}[index_zm]) / dx'
        else:
            return f'({field}[index_zp] - {field}[index_zm]) / (2 * dx)'
    elif deriv == 3:
        if boundaryFlags[0]:
            return f'({field}[index_xpp] - 2 * {field}[index_xp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[1]:
            return f'({field}[index] - 2 * {field}[index_xm] + {field}[index_xmm]) / (dx * dx)'
        else:
            return f'({field}[index_xp] - 2 * {field}[index] + {field}[index_xm]) / (dx * dx)'
    elif deriv == 4:
        if boundaryFlags[2]:
            return f'({field}[index_ypp] - 2 * {field}[index_yp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[3]:
            return f'({field}[index] - 2 * {field}[index_ym] + {field}[index_ymm]) / (dx * dx)'
        else:
            return f'({field}[index_yp] - 2 * {field}[index] + {field}[index_ym]) / (dx * dx)'
    elif deriv == 5:
        if boundaryFlags[4]:
            return f'({field}[index_zpp] - 2 * {field}[index_zp] + {field}[index]) / (dx * dx)'
        elif boundaryFlags[5]:
            return f'({field}[index] - 2 * {field}[index_zm] + {field}[index_zmm]) / (dx * dx)'
        else:
            return f'({field}[index_zp] - 2 * {field}[index] + {field}[index_zm]) / (dx * dx)'

def getVelocityUpdateU(boundaryFlags):
    if boundaryFlags[0] or boundaryFlags[1]:
        return 'f2.u[index] = 0;'
    else:
        return f"""
    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * {getDeriv('f1.u', 0, boundaryFlags)}
        - f1.v[index] * {getDeriv('f1.u', 1, boundaryFlags)}
        - f1.w[index] * {getDeriv('f1.u', 2, boundaryFlags)}
        - (RT/f1.p[index]) * {getDeriv('f1.p', 0, boundaryFlags)}
        + mu * (RT/f1.p[index]) * (
            {getDeriv('f1.u', 3, boundaryFlags)} +
            {getDeriv('f1.u', 4, boundaryFlags)} +
            {getDeriv('f1.u', 5, boundaryFlags)}
        )
    );
    """

def getVelocityUpdateV(boundaryFlags):
    if boundaryFlags[2] or boundaryFlags[3]:
        return 'f2.v[index] = 0;'
    else:
        return f"""
    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * {getDeriv('f1.v', 0, boundaryFlags)}
        - f1.v[index] * {getDeriv('f1.v', 1, boundaryFlags)}
        - f1.w[index] * {getDeriv('f1.v', 2, boundaryFlags)}
        - (RT/f1.p[index]) * {getDeriv('f1.p', 1, boundaryFlags)}
        + mu * (RT/f1.p[index]) * (
            {getDeriv('f1.v', 3, boundaryFlags)} +
            {getDeriv('f1.v', 4, boundaryFlags)} +
            {getDeriv('f1.v', 5, boundaryFlags)}
        )
    );
    """

def getVelocityUpdateW(boundaryFlags):
    if boundaryFlags[4] or boundaryFlags[5]:
        return 'f2.w[index] = 0;'
    else:
        return f"""
    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * {getDeriv('f1.w', 0, boundaryFlags)}
        - f1.v[index] * {getDeriv('f1.w', 1, boundaryFlags)}
        - f1.w[index] * {getDeriv('f1.w', 2, boundaryFlags)}
        - (RT/f1.p[index]) * {getDeriv('f1.p', 2, boundaryFlags)}
        + mu * (RT/f1.p[index]) * (
            {getDeriv('f1.w', 3, boundaryFlags)} +
            {getDeriv('f1.w', 4, boundaryFlags)} +
            {getDeriv('f1.w', 5, boundaryFlags)}
        )
        - g
    );
    """

def getKernelBody(boundaryFlags):
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
    {getVelocityUpdateU(boundaryFlags)}
    {getVelocityUpdateV(boundaryFlags)}
    {getVelocityUpdateW(boundaryFlags)}
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        {getDeriv('f1.u', 0, boundaryFlags)} * f1.p[index] +
        {getDeriv('f1.p', 0, boundaryFlags)} * f1.u[index] +
        {getDeriv('f1.v', 1, boundaryFlags)} * f1.p[index] +
        {getDeriv('f1.p', 1, boundaryFlags)} * f1.v[index] +
        {getDeriv('f1.w', 2, boundaryFlags)} * f1.p[index] +
        {getDeriv('f1.p', 2, boundaryFlags)} * f1.w[index]
    );
    """
    kernel_body = textwrap.dedent(kernel_body)    
    kernel_body = textwrap.indent(kernel_body, ' ' * 8)
    return kernel_body

def generateCoreKernel(boundaryFlags):
    coreSide = side - 2
    coreSize = coreSide ** 3
    kernel = ''
    kernel += f"""
    __global__ void updateCore(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(totalThreadIndex >= {coreSize}) return;

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

        {getKernelBody(boundaryFlags)}
    }}
    """
    return kernel

def getIndexExpressions(side, boundaryFlags):
    initialExpressions = [
        f'int index_xm = (x-1) + y * {side} + z * {side} * {side};',
        f'int index_xp = (x+1) + y * {side} + z * {side} * {side};',
        f'int index_ym = x + (y-1) * {side} + z * {side} * {side};',
        f'int index_yp = x + (y+1) * {side} + z * {side} * {side};',
        f'int index_zm = x + y * {side} + (z-1) * {side} * {side};',
        f'int index_zp = x + y * {side} + (z+1) * {side} * {side};']
    for i, flag in enumerate(boundaryFlags):
        if flag:
            if i % 2 == 0:
                initialExpressions[i] = initialExpressions[i].replace('m', 'pp')
                initialExpressions[i] = initialExpressions[i].replace('-1', '+2')
            else:
                initialExpressions[i] = initialExpressions[i].replace('p', 'mm')
                initialExpressions[i] = initialExpressions[i].replace('+1', '-2')
    finalExpressions = '\n'.join(initialExpressions)
    return textwrap.indent(finalExpressions, ' ' * 8)

def getFaceIndexing(face, side, faceSide, boundaryFlags):
    if face[0] == 'X':
        return f"""
        int x = {0 if face[1] == '1' else side - 1};
        int y = (totalThreadIndex % {faceSide}) + 1;
        int z = ((totalThreadIndex / {faceSide}) % {faceSide}) + 1;
        int index = x + y * {side} + z * {side} * {side};
{getIndexExpressions(side, boundaryFlags)}
        """
    elif face[0] == 'Y':
        return f"""
        int x = (totalThreadIndex % {faceSide}) + 1;
        int y = {0 if face[1] == '1' else side - 1};
        int z = ((totalThreadIndex / {faceSide}) % {faceSide}) + 1;
        int index = x + y * {side} + z * {side} * {side};
{getIndexExpressions(side, boundaryFlags)}
        """
    elif face[0] == 'Z':
        return f"""
        int x = (totalThreadIndex % {faceSide}) + 1;
        int y = ((totalThreadIndex / {faceSide}) % {faceSide}) + 1;
        int z = {0 if face[1] == '1' else side - 1};
        int index = x + y * {side} + z * {side} * {side};
{getIndexExpressions(side, boundaryFlags)}
        """

def generateFaceKernel(face, boundaryFlags):
    faceSide = side - 2
    faceSize = faceSide ** 2
    kernel = ''
    kernel += f"""
    __global__ void updateFace{face}(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(totalThreadIndex >= {faceSize}) return;

        {getFaceIndexing(face, side, faceSide, boundaryFlags)}

        {getKernelBody(boundaryFlags)}
    }}
    """
    return kernel

def getEdgeIndexing(edge, side, boundaryFlags):
        # Compute x, y, z
        if 'X' in edge:
            val = 0 if 'X1' in edge else side - 1
            x_line = f'int x = {val};'
        else:
            x_line = f'int x = totalThreadIndex + 1;'
        if 'Y' in edge:
            val = 0 if 'Y1' in edge else side - 1
            y_line = f'int y = {val};'
        else:
            y_line = f'int y = totalThreadIndex + 1;'
        if 'Z' in edge:
            val = 0 if 'Z1' in edge else side - 1
            z_line = f'int z = {val};'
        else:
            z_line = f'int z = totalThreadIndex + 1;'
        return f"""
        {x_line}
        {y_line}
        {z_line}
        int index = x + y * {side} + z * {side} * {side};
{getIndexExpressions(side, boundaryFlags)}
        """

def generateEdgeKernel(edge, boundaryFlags):
    edgeSize = side - 2
    kernel = ''
    kernel += f"""
    __global__ void updateEdge{edge}(data d) {{
        int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if(totalThreadIndex >= {edgeSize}) return;

        {getEdgeIndexing(edge, side, boundaryFlags)}

        {getKernelBody(boundaryFlags)}
    }}
    """
    return kernel

def getCornerIndexing(corner, side, boundaryFlags):
    if 'X1' in corner:
        x_line = 'int x = 0;'
    else:
        x_line = f'int x = {side} - 1;'
    if 'Y1' in corner:
        y_line = 'int y = 0;'
    else:
        y_line = f'int y = {side} - 1;'
    if 'Z1' in corner:
        z_line = 'int z = 0;'
    else:
        z_line = f'int z = {side} - 1;'
    return f"""
        {x_line}
        {y_line}
        {z_line}
        int index = x + y * {side} + z * {side} * {side};
{getIndexExpressions(side, boundaryFlags)}
    """

def generateCornerKernel(corner, boundaryFlags):
    kernel = ''
    kernel += f"""
    __global__ void updateCorner{corner}(data d) {{
        {getCornerIndexing(corner, side, boundaryFlags)}

        {getKernelBody(boundaryFlags)}
    }}
    """
    return kernel

def generateListofKernels(cases):
    kernels = []
    for case in cases:
        if case[0] == 'CORE':
            newKernel = generateCoreKernel(case[1])
            newKernel = textwrap.dedent(newKernel)
            kernels.append(newKernel)
        elif len(case[0]) == 2:
            # Face
            newKernel = generateFaceKernel(*case)
            newKernel = textwrap.dedent(newKernel)
            kernels.append(newKernel)
        elif len(case[0]) == 4:
            # Edge
            newKernel = generateEdgeKernel(*case)
            newKernel = textwrap.dedent(newKernel)
            kernels.append(newKernel)
        elif len(case[0]) == 6:
            # Corner
            newKernel = generateCornerKernel(*case)
            newKernel = textwrap.dedent(newKernel)
            kernels.append(newKernel)
    return kernels

def generateListofCalls(cases):
    calls = []
    coreBlocks = (side - 2) ** 3 // threadsPerBlock + 1
    faceBlocks = (side - 2) ** 2 // threadsPerBlock + 1
    edgeBlocks = (side - 2) // threadsPerBlock + 1
    for case in cases:
        if case[0] == 'CORE':
            calls.append(f'updateCore<<<{coreBlocks}, {threadsPerBlock}>>>(d);')
        elif len(case[0]) == 2:
            calls.append(f'    updateFace{case[0]}<<<{faceBlocks}, {threadsPerBlock}>>>(d);')
        elif len(case[0]) == 4:
            calls.append(f'    updateEdge{case[0]}<<<{edgeBlocks}, {threadsPerBlock}>>>(d);')
        elif len(case[0]) == 6:
            calls.append(f'    updateCorner{case[0]}<<<1, 1>>>(d);')
    return calls

def preCompile():
    # This function is called before building the CUDA/C++ code
    # It is used to automatically generate high-performance but (directly) unmaintainable code to handle CFD updates in the cube's core,
    # faces, edges, and corners.
    cases = generateListOfCases()
    kernels = generateListofKernels(cases)
    calls = generateListofCalls(cases)

    with open('../src/kernel.cu', 'r') as f:
        kernelsFileContent = f.read()
    newContent = kernelsFileContent.replace('// *** PYTHON CODE-GENERATED KERNEL DEFINITIONS ***', ''.join(kernels))
    newContent = newContent.replace('// *** PYTHON CODE-GENERATED KERNEL CALLS ***', '\n'.join(calls))
    with open('../src/kernel_w.cu', 'w') as f:
        f.write(newContent)

    with open('../src/main.cpp', 'r') as f:
        mainFileContent = f.read()
    newContent = mainFileContent.replace('// *** PYTHON CODE-GENERATED SIDE VALUE ***', f'constexpr int side = {side};')
    with open('../src/main_w.cpp', 'w') as f:
        f.write(newContent)


preCompile()