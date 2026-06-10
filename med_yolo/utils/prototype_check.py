import torch

# 替换为你实际保存的路径
proto_path = '/media/Storage3/wmm/ICML/Medical-OD/work_dirs/prototypes.pt' 
data = torch.load(proto_path)

print("Keys:", data.keys())
print("Class IDs:", data['class_ids'])

# 检查视觉原型形状
# 预期形状: [类别数, 512]
if 'vis_prototypes' in data:
    print("Visual Prototypes Shape:", data['vis_prototypes'].shape)

# 检查几何原型形状
# 预期形状: [类别数, 4] (w, h, r, a)
if 'geo_prototypes' in data:
    print("Geometric Prototypes Shape:", data['geo_prototypes'].shape)