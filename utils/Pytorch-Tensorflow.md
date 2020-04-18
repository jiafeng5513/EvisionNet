### Migration Guide for TensorFlow To PyTorch

|序号|TensorFlow|PyTorch|备注|
|:---:|:---:|:---:|:---:|
|1| tf.expand_dims(x, 1)|torch.unsqueeze(x, 1)||
|2| tf.squeeze | torch.squeeze||
|3| tf.einsum | torch.einsum ||
|4| tf.transpose(rot_mat, [0, 3, 4, 1, 2])|rot_mat.permute(0, 3, 4, 1, 2)||
|5| tf.unstack(x,dim=1)|torch.unbind(x,dim=1)||
|6| tf.is_nan() |torch.isnan()||
|7| tf.logical_and(x,y) | x.mul(y).bool()||
|8| tf.clip_by_value(x, min, max) | torch.clamp(input,min,max) ||
|9| tf.reduce_all(x, axis=0) | x.prod(dim=0).bool() ||
|10| tf.rank(x) | len(x.shape) ||
|11| tf.matmul(x,y) |torch.matmul(x,y)||
|12| tf.map_fn()| ---- ||
|13| tf.greater(x, y) | torch.gt(x,y) ||
|14| tf.reduce_mean() |||
|15| tf.reduce_any(x,i) | x.sum(i).bool() ||
|15| tf.reduce_sum(x,i) | x.sum(i) ||
|16| tf.concat(x,i)|torch.cat(x.i)||
|18| tf.multiply |*或者torch.mul||
|19|tf.nn.l2_normalize(x, 1)|sklearn.preprocessing.normalize(x , norm='l2')||
|20|tf.multinomial(x, 1)|torch.multinomial(x, 1, replacement=True)||