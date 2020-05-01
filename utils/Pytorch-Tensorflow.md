### Migration Guide for TensorFlow To PyTorch

|序号|TensorFlow|PyTorch|备注|
|:---:|:---:|:---:|:---:|
| 1| tf.expand_dims(x, 1)|torch.unsqueeze(x, 1)|沿着指定位置扩展维度|
| 2| tf.squeeze | torch.squeeze|去除等于1的维度|
| 3| tf.unstack(x,dim=1)|torch.unbind(x,dim=1)|沿着指定的维度拆开张量|
| 4| tf.concat(x,i)|torch.cat(x.i)|沿着指点的维度链接|
| 5| tf.is_nan() |torch.isnan()|判断是否为nan|
| 6| tf.logical_and(x,y) | x.mul(y).bool()|逻辑与|
| 7| tf.greater(x, y) | torch.gt(x,y) |返回一个x>y的位置为true的张量|
| 8|z=tf.less(x,y)|z=torch.le(x,y)|返回一个x<y的位置为true的张量|
| 9| tf.matmul(x,y) |torch.matmul(x,y)|矩阵乘法(乘加)|
|10| tf.multiply |*或者torch.mul|数乘和元素乘|
|11|tf.square(x)|x.pow(w)|元素平方(N次方)|
|12| tf.clip_by_value(x, min, max) | torch.clamp(input,min,max) |上下限截断|
|13| tf.rank(x) | len(x.shape) |长度|
|14| tf.transpose(x, [0, 3, 4, 1, 2])|x.permute(0, 3, 4, 1, 2)|维度换位|
|15| tf.reduce_mean(x,axis=[1,2], keepdims=False)|torch.mean(x, dim=[1,2], keepdim=False)|均值|
|16| tf.reduce_any(x,axis=i,keepdims=False) | x.sum(dim=i,keepdim=False).bool() |沿指定维度连续逻辑或|
|18| tf.reduce_all(x, axis=0) | x.prod(dim=0).bool() |沿指定维度连续逻辑与|
|17| tf.reduce_sum(x,i) | x.sum(i) |求和|
|19|tf.nn.l2_normalize(x, 1)|sklearn.preprocessing.normalize(x , norm='l2')|l2函数|
|20|tf.multinomial(x, 1)<br>(新的函数名为tf.random.categorical)|torch.multinomial(x, 1, replacement=True)|随机抽样|
|21|z = tf.broadcast_to(x, y.shape)|z, _ = torch.broadcast_tensors(x, y)|显式广播语义|
|22| tf.einsum | torch.einsum |爱因斯坦和|
