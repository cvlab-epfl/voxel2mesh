import torch
from scipy.spatial import ConvexHull
from itertools import combinations
def get_commont_vertex(edge_pair):
    a = edge_pair[:, 0] == edge_pair[:, 1]
    b = edge_pair[:, 0] == torch.flip(edge_pair[:, 1], dims=[1])

    return edge_pair[:, 0][a + b]

def uniform_unpool(vertices_, faces_, identical_face_batch=True):
    if vertices_ is None:
        return None, None
    batch_size , _, _ = vertices_.shape
    new_faces_all = []
    new_vertices_all = []

    for vertices, faces in zip(vertices_, faces_):
        face_count, _ = faces.shape
        vertices_count = len(vertices)
        edge_combinations_3 = torch.tensor(list(combinations(range(3), 2)))
        edges = faces[:, edge_combinations_3]
        unique_edges = edges.view(-1, 2)
        unique_edges, _ = torch.sort(unique_edges, dim=1)
        unique_edges, unique_edge_indices = torch.unique(unique_edges, return_inverse=True, dim=0)
        face_edges = vertices[unique_edges]

        ''' Computer new vertices '''
        new_vertices = torch.mean(face_edges, dim=1)
        new_vertices = torch.cat([vertices, new_vertices], dim=0)  # <----------------------- new vertices + old vertices
        new_vertices_all += [new_vertices[None]]

        ''' Compute new faces '''
        corner_faces = []
        middle_face = []
        for j, combination in enumerate(edge_combinations_3):
            edge_pair = edges[:, combination]
            common_vertex = get_commont_vertex(edge_pair)

            new_vertex_1 = unique_edge_indices[torch.arange(0, 3 * face_count, 3) + combination[0]] + vertices_count
            new_vertex_2 = unique_edge_indices[torch.arange(0, 3 * face_count, 3) + combination[1]] + vertices_count

            middle_face += [new_vertex_1[:, None], new_vertex_2[:, None]]
            corner_faces += [torch.cat([common_vertex[:, None], new_vertex_1[:, None], new_vertex_2[:, None]], dim=1)]

        corner_faces = torch.cat(corner_faces, dim=0)
        middle_face = torch.cat(middle_face, dim=1)
        middle_face = torch.unique(middle_face, dim=1)
        new_faces_all += [torch.cat([corner_faces, middle_face], dim=0)[None]]  # new faces-3

        if identical_face_batch:
            new_vertices_all = new_vertices_all[0].repeat(batch_size, 1, 1)
            new_faces_all = new_faces_all[0].repeat(batch_size, 1, 1)
            break

    return new_vertices_all, new_faces_all


def adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, N_prev):
    vertices_primary = vertices[0,:N_prev, :]
    vertices_secondary = vertices[0,N_prev:, :]
    faces_primary = faces_prev[0]
    
    sphere_vertices_primary = sphere_vertices[0,:N_prev]
    sphere_vertices_secondary = sphere_vertices[0,N_prev:]

    if latent_features is not None:
        latent_features_primary = latent_features[0,:N_prev]
        latent_features_secondary = latent_features[0,N_prev:]

    face_count, _ = faces_primary.shape
    vertices_count = len(vertices_primary)
    edge_combinations_3 = torch.tensor(list(combinations(range(3), 2))).cuda()
    edges = faces_primary[:, edge_combinations_3]
    unique_edges = edges.view(-1, 2)
    unique_edges, _ = torch.sort(unique_edges, dim=1)
    unique_edges, unique_edge_indices = torch.unique(unique_edges, return_inverse=True, dim=0)
    face_edges_primary = vertices_primary[unique_edges]

    a = face_edges_primary[:,0]
    b = face_edges_primary[:,1]
    v = vertices_secondary

    va = v - a
    vb = v - b
    ba = b - a

    cond1 = (va * ba).sum(1)
    norm1 = torch.norm(va, dim=1)

    cond2 = (vb * ba).sum(1)
    norm2 = torch.norm(vb, dim=1)

    dist = torch.norm(torch.cross(va, ba), dim=1)/torch.norm(ba, dim=1)
    dist[cond1 < 0] = norm1[cond1 < 0]
    dist[cond2 < 0] = norm2[cond2 < 0]

    sorted_, _ = torch.sort(dist)
    threshold = sorted_[int(0.3*len(sorted_))] 

    vertices_needed = vertices_secondary[dist > threshold]
    
    sphere_vertices_needed = sphere_vertices_secondary[dist > threshold] 
    if latent_features is not None:
        latent_features_needed = latent_features_secondary[dist > threshold]

    vertices = torch.cat([vertices_primary,vertices_needed],dim=0)[None]
    if latent_features is not None:
        latent_features = torch.cat([latent_features_primary,latent_features_needed],dim=0)[None]

    sphere_vertices = torch.cat([sphere_vertices_primary,sphere_vertices_needed],dim=0) 
    sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])
    hull = ConvexHull(sphere_vertices.data.cpu().numpy())  
    faces = torch.from_numpy(hull.simplices).long().cuda()[None] 

    sphere_vertices = sphere_vertices[None]  

    return vertices, faces, latent_features, sphere_vertices

