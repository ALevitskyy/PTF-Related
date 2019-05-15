vertice_list = []
ob = D.objects.get("Armature").children[0]
bpy.context.scene.objects.active = ob
for face in ob.data.polygons:
    for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
        uv_coords =ob.data.uv_layers.active.data[loop_idx].uv
        vertex = {"faceidx":face.index,
                  "vertidx":vert_idx,
                   "u":uv_coords.x,
                   "v":uv_coords.y}
        vertice_list.append(vertex)
#print("face idx: %i, vert idx: %i, uvs: %f, %f" % (face.index,
#                           vert_idx, uv_coords.x, uv_coords.y))

import pickle
faces = {}
for face in bm.faces:
    face_list = []
    for i in range(3):
        vertex = {str(face.verts[i].index):{"coordinates":np.array(face.verts[i].co),
                                       "uvs":np.array(face.loops[i][bm.loops.layers.uv[0]].uv),
                                        "normal":np.array(face.verts[i].normal)}}
        face_list.append(vertex)
    faces[str(face.index)] = face_list
pickle.dump(faces, open("/Users/andriylevitskyy/Desktop/blender.pkl","wb"))
