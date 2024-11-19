import openmesh as om
import polyscope as ps
import numpy as np
import matplotlib.colors as mcolors
import sys

class Smoother:
    def __init__(self, mesh):
        """
        Initialize the Smoother class with an OpenMesh mesh.
        
        Parameters:
        - mesh: An instance of OpenMesh PolyMesh
        """
        self.mesh = mesh
        print("computing normals...")
        self.mesh.update_vertex_normals()
        # storing normals for tangential smoothing
        self.normals = mesh.vertex_normals()
        print("...done")

        # storing points in numpy array for processing outside OpenMesh
        self.points = self.mesh.points()

        # place holders for the quantities to compute in the practical
        self.uniformLaplaceVector = np.zeros( (self.mesh.n_vertices(), 3))
        self.LaplaceBeltramiVector = np.zeros( (self.mesh.n_vertices(), 3))

        self.uniformMeanCurvature = np.zeros(self.mesh.n_vertices())
        self.LaplaceBeltramiMeanCurvature = np.zeros(self.mesh.n_vertices())
        self.GaussCurvature = np.zeros(self.mesh.n_vertices())
        self.triangleQuality = np.zeros( self.mesh.n_faces())

    # PRECOMPUTE THE LIST OF ONE-RING VERTEX INDICES
    def initVertexRingVertex( self):
        print("init vertex indices of vertex ring...")
        self.vertexRingVertexIndices = []
        for vh in self.mesh.vertices():
            self.vertexRingVertexIndices.append([vv.idx() for vv in self.mesh.vv(vh)])
        print("...done")

    # PRECOMPUTE THE LIST OF ONE-RING EDGE INDICES
    def initVertexRingEdge( self):
        print("init edge indices of vertex ring...")
        self.vertexRingEdgeIndices = []
        for vh in self.mesh.vertices():
            self.vertexRingEdgeIndices.append([ve.idx() for ve in self.mesh.ve(vh)])
        print("...done")

    def computeTriangleQuality(self):
        print("Computing triangle quality...")

        # Iterate over all triangles in the mesh
        for fi, f in enumerate(self.mesh.faces()):
            # Get the vertices of the triangle
            v0, v1, v2 = [vh.idx() for vh in self.mesh.fv(f)]

            # Compute vectors a, b, c
            a = self.points[v1] - self.points[v0]
            b = self.points[v2] - self.points[v0]
            c = self.points[v2] - self.points[v1]


            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            c_norm = np.linalg.norm(c)

            # Compute the product of the magnitudes |a| * |b| * |c|
            product_of_magnitudes = a_norm * b_norm * c_norm
            # Compute the area using the cross product of a and b
            cross_product = np.cross(a, b)
            
            if np.linalg.norm(cross_product) == 0:
                circumradius = 0
            else :circumradius = product_of_magnitudes / (2 * np.linalg.norm(cross_product))

            
            min_edge_length = min(a_norm, b_norm, c_norm)
            if min_edge_length == 0:
                self.triangleQuality[fi] = 0
                continue
            # Calculate the triangle quality ratio
            else:self.triangleQuality[fi] = circumradius / min_edge_length

        print("...done")



    def computeUniformLaplaceOperator(self):
        print("computing uniform Laplace operator...")
        

            # Iterate over all the vertices in the mesh
        for vi, vh in enumerate(self.vertexRingVertexIndices):
            # Initialize the sum of neighbors' positions as a zero vector
            sum_neighbors = np.zeros_like(self.points[vi])

            # Calculate the sum of positions of neighbors
            # Sum vi
            for neighbor_idx in vh:  
                neighbor_position = self.points[neighbor_idx]  # Use points array to get vertex position
                sum_neighbors += neighbor_position
            
            # 1/n * Sum(vi)
            # Calculate the average position of the neighbors
            average_position = sum_neighbors / len(vh)


            # Get the current position of the vertex
            #v
            current_position = self.points[vi]

            # Calculate the Laplacian vector: L(v_i) = average_position - current_position
            # L(v_i) = 1/n * Sum(vi) - v
            Laplace_vector = average_position - current_position

            # Store the Laplacian vector for the current vertex
            self.uniformLaplaceVector[vi] = Laplace_vector


        print("...done")
            
    def computeUniformMeanCurvature(self):
        print("compute uniform mean curvature...")
        self.computeUniformLaplaceOperator()
        for i in range(self.mesh.n_vertices()):
            self.uniformMeanCurvature[i]  = (-1/2)*np.linalg.norm(self.uniformLaplaceVector[i])
        print("...done")

    def uniformLaplaceSmoothing( self, nbIter):
        for i in range( nbIter):
            print("compute one uniform smoothing step...")
            self.computeUniformLaplaceOperator()
            for vi in range( self.mesh.n_vertices()):
                self.points[vi] += 0.5 * self.uniformLaplaceVector[vi]

            print("...done")

    def tangentialSmoothing(self, nbIter):
        for i in range(nbIter):
            print(f"Compute one tangential smoothing step... Iteration {i + 1}/{nbIter}")
            
            # Step 1: Compute uniform Laplace operator
            self.computeUniformLaplaceOperator()

            # Step 2: Perform tangential smoothing
            for vi in range(self.mesh.n_vertices()):
                # Get the uniform Laplace vector and the normal for this vertex
                laplace_vector = self.uniformLaplaceVector[vi]
                normal = self.normals[vi]

                # Project the Laplace vector onto the tangent plane
                tangential_laplace = laplace_vector - np.dot(laplace_vector, normal) * normal

                # Update the vertex position
                self.points[vi] += 0.5 * tangential_laplace  # 0.5 is a smoothing factor, can be adjusted

        print("...done")
            
    def initCoTangentEdgeVertexIndices(self):
        print("computing cotangent edge vertex indices...")
        self.coTangentEdgeVertexIndices = []
        for e in self.mesh.edges():
            h0 = self.mesh.halfedge_handle( e, 0)
            v0 = self.mesh.to_vertex_handle( h0).idx()
            h1 = self.mesh.halfedge_handle( e, 1)
            v1 = self.mesh.to_vertex_handle( h1).idx()
            h2 = self.mesh.next_halfedge_handle( h0)
            v2 = self.mesh.to_vertex_handle( h2).idx()
            h3 = self.mesh.next_halfedge_handle( h1)
            v3 = self.mesh.to_vertex_handle( h3).idx()
            self.coTangentEdgeVertexIndices.append( [ v0, v1, v2, v3])
        print("...done")

    def computeCoTangentWeights( self):
        print("computing cotangent weights...")
        self.coTangentWeight = np.zeros( self.mesh.n_edges())
        for ei in range( self.mesh.n_edges()):
            w = 0.
            v0, v1, v2, v3 = self.coTangentEdgeVertexIndices[ ei]
            ctP = self.points[[v0, v1, v2, v3]]
            for idx in [2,3]:
                d0 = ctP[0] - ctP[idx]
                d1 = ctP[1] - ctP[idx]
                d0 = d0 / np.linalg.norm(d0)
                d1 = d1 / np.linalg.norm(d1)
                w += 1.0 / np.tan( np.acos( min( 0.99, max( -0.99, np.dot( d0, d1)))))
            w = max( 0., w)
            self.coTangentWeight[ ei] = w
        print("...done")


    # def computeLaplaceBeltramiOperator(self):
    #     print("computing Laplace Beltrami operator...")
    #     self.computeCoTangentWeights()

    #     # Iterate through each vertex and its corresponding edges
    #     for vi, edge_indices in enumerate(self.vertexRingEdgeIndices):
    #         current_position = self.points[vi]

    #         # Iterate through each edge connected to the current vertex
    #         for ei in edge_indices:

    #             v0, v1, v2, v3 = self.coTangentEdgeVertexIndices[ ei]
    #             if v0 == vi:
    #                 neighbor_idx = v1
    #             elif v1 == vi:
    #                 neighbor_idx = v0
    #             else:
    #                 continue

    #             weight = self.coTangentWeight[ei]

    #             self.LaplaceBeltramiVector[vi] += weight * (self.points[neighbor_idx] - current_position)

    #     print("...done")



    def computeLaplaceBeltramiOperator(self):
        print("computing Laplace Beltrami operator...")

        for vi in range(self.mesh.n_vertices()):
            acc = np.zeros(3)
            for e in self.vertexRingEdgeIndices[vi]:
                v0, v1, v2, v3 = self.coTangentEdgeVertexIndices[e]
                if vi == v0:
                    acc += self.coTangentWeight[e] * (self.points[v1] - self.points[vi])
                    
                elif vi == v1:
                    acc += self.coTangentWeight[e] * (self.points[v0] - self.points[vi])
                    
            self.LaplaceBeltramiVector[vi] = acc

        print("...done")

    def computeLaplaceBeltramiMeanCurvature(self):
        print("computing Laplace-Beltrami mean curvature...")
        # Compute the Laplace-Beltrami vectors
        self.computeLaplaceBeltramiOperator()

        # Compute the mean curvature for each vertex
        for vi in range(self.mesh.n_vertices()):
            laplace_magnitude = np.linalg.norm(self.LaplaceBeltramiVector[vi])
            self.LaplaceBeltramiMeanCurvature[vi] = 0.5 * laplace_magnitude

        

        print("...done")



    def LaplaceBeltramiSmoothing(self, nbIter):
        for i in range(nbIter):
            print(f"Computing one Laplace-Beltrami smoothing step... Iteration {i + 1}/{nbIter}")

            self.computeLaplaceBeltramiOperator()
            self.computeCoTangentWeights()
            # new_lb = 1/sum(wi) * Sum(wi * (vi - v))
            # v' = v + 0.5 * new_lb
            for vi in range(self.mesh.n_vertices()):
                weight_sum = 0.0
                new_lb = np.zeros(3)
                for e in self.vertexRingEdgeIndices[vi]:
                    weight_sum += self.coTangentWeight[e]
                new_lb = (1 / (weight_sum)) * self.LaplaceBeltramiVector[vi]
                self.points[vi] += 0.5 * new_lb


            print("...done")


    def computeGaussCurvature(self):
        print("compute Gauss curvature...")

        # Initialize the Gaussian curvature array
        self.GaussCurvature = np.zeros(self.mesh.n_vertices())

        # Loop through each vertex
        for vi in range(self.mesh.n_vertices()):
            angle_sum = 0.0

            # Get the neighboring faces of the vertex
            for fh in self.mesh.vf(self.mesh.vertex_handle(vi)):  # Iterate over adjacent faces
                # Get the vertices of the face
                face_vertices = [vh.idx() for vh in self.mesh.fv(fh)]
                # Ensure we calculate the angle for the current vertex
                if vi in face_vertices:
                    v0, v1, v2 = [self.points[idx] for idx in face_vertices]
                    if vi == face_vertices[0]:
                        a, b, c = v1, v2, v0
                    elif vi == face_vertices[1]:
                        a, b, c = v2, v0, v1
                    else:
                        a, b, c = v0, v1, v2

                    # Compute the angle at vertex vi using the cosine rule
                    ba = a - b
                    bc = c - b
                    ba /= np.linalg.norm(ba)
                    bc /= np.linalg.norm(bc)
                    angle = np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))
                    angle_sum += angle

            # Compute Gaussian curvature as the angle defect
            self.GaussCurvature[vi] = 2 * np.pi - angle_sum

        print("...done")


def scalarFieldToColors( scalarField, minC=None, maxC=None):
    """
    Convert a scalar field to HSV colors, where each scalar value is mapped to a unique color.
    
    Parameters:
    - scalar_field: A 1D numpy array of scalar values.
    - minC, maxC: clamping values (optional)
    
    Returns:
    - colors: A Nx3 numpy array of RGB colors corresponding to each scalar value.
    """
    if minC != None:
        assert(maxC!=None)
        minVal = minC
        maxVal = maxC
    else:
        minVal = np.min( scalarField)
        maxVal = np.max( scalarField)

    np.clip( scalarField, minVal, maxVal, out=scalarField)
    if (maxVal - minVal) > 1.e-20:
        normalized_field = (scalarField - minVal) / (maxVal - minVal)
        hsv_field = 2./3. * ( 1. - normalized_field) # min <-> blue (cold), max <-> red (hot)
    else:
        hsv_field = 1./3. * np.ones( scalarField.shape)

    # Map each normalized value to a hue in the HSV space
    # We use HSV (Hue, Saturation, Value) where:
    # - Hue varies with the scalar value
    # - Saturation and Value are fixed to 1 for vibrant colors
    reshaped_array = hsv_field.reshape(-1, 1)
    # Add two columns of 1s
    hsv_colors = np.hstack((reshaped_array, np.ones((reshaped_array.shape[0], 2))))
    # hsv_colors = np.array([[h, 1.0, 1.0] for h in normalized_field])
    # Convert HSV to RGB
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    return rgb_colors   

def visualize_mesh( mesh, vScalarField, fScalarField):
    # Prepare the data for visualization
    print("mapping to polyscope arrays...")
    vertices = mesh.points() # array of vertex positions
    faces = mesh.face_vertex_indices()  # array of face vertex indices
    vColors = scalarFieldToColors( vScalarField)
    fColors = scalarFieldToColors( fScalarField, minC=0.6, maxC=2.0)
    print("...done")

    # Initialize Polyscope
    ps.init()
        
    # Register the mesh with Polyscope
    ps_mesh = ps.register_surface_mesh("My Mesh", vertices, faces)

    ps_mesh.add_color_quantity("scalar field on vertices", vColors, defined_on='vertices')
    ps_mesh.add_color_quantity("scalar field on faces", fColors, defined_on='faces')

    # Show the mesh in the Polyscope viewer
    ps.show()

def main():
    if len(sys.argv) != 3:
        print( f"usage: {sys.argv[0]} <meshFileName> <nbIter>")
        sys.exit(1)
    fileName = sys.argv[1]
    nbIter = int(sys.argv[2])
    mesh = om.read_polymesh( fileName)
    smootherMesh = Smoother(mesh)

    # INITIALIZE INDICES
    smootherMesh.initVertexRingVertex()
    smootherMesh.initVertexRingEdge()
    smootherMesh.initCoTangentEdgeVertexIndices()
    smootherMesh.computeCoTangentWeights()

    # SMOOTHING
    # smootherMesh.uniformLaplaceSmoothing(  nbIter)
    # smootherMesh.tangentialSmoothing(  nbIter)
    smootherMesh.LaplaceBeltramiSmoothing( nbIter)

    # ANALYZING
    # smootherMesh.computeUniformMeanCurvature()
    # smootherMesh.computeLaplaceBeltramiMeanCurvature()
    smootherMesh.computeGaussCurvature()
    smootherMesh.computeTriangleQuality()

    outputMesh = om.TriMesh( smootherMesh.points, mesh.face_vertex_indices())
    # visualize_mesh( outputMesh, smootherMesh.uniformMeanCurvature, smootherMesh.triangleQuality)
    # visualize_mesh( outputMesh, smootherMesh.LaplaceBeltramiMeanCurvature, smootherMesh.triangleQuality)
    visualize_mesh( outputMesh, smootherMesh.GaussCurvature, smootherMesh.triangleQuality)
    return None

if __name__=="__main__":
    main()
