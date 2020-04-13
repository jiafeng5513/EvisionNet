# -*- coding: utf-8 -*-
"""Helper functions for geometric transforms."""

import torch


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
    """Builds a rotation matrix from sines and cosines of Euler angles.

Note:
  In the following, A1 to An are optional batch dimensions.

Args:
  sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
    represents the sine of the Euler angles.
  cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
    represents the cosine of the Euler angles.

Returns:
  A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
  represent a 3d rotation matrix.
"""
    # sin_angles.shape.assert_is_compatible_with(cos_angles.shape)
    sx, sy, sz = torch.unbind(sin_angles, dim=-1)
    cx, cy, cz = torch.unbind(cos_angles, dim=-1)
    m00 = cy * cz
    m01 = (sx * sy * cz) - (cx * sz)
    m02 = (cx * sy * cz) + (sx * sz)
    m10 = cy * sz
    m11 = (sx * sy * sz) + (cx * cz)
    m12 = (cx * sy * sz) - (sx * cz)
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    matrix = torch.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=-1)  # pyformat: disable
    output_shape = torch.cat((sin_angles.shape[:-1], (3, 3)), dim=-1)
    return matrix.reshape(output_shape)


def from_euler(angles):
    r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """

    angles = torch.Tensor(angles)
    # shape.check_static(tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def matrix_from_angles(rot):
    """
      Create a rotation matrix from a triplet of rotation angles.
      Args:
        rot: a tf.Tensor of shape [..., 3], where the last dimension is the rotation angles, along x, y, and z.

      Returns:
        A tf.tensor of shape [..., 3, 3], where the last two dimensions are the rotation matrix.

      This function mimics _euler2mat from struct2depth/project.py, for backward compatibility,
      but wraps tensorflow_graphics instead of reimplementing it.
      The negation and transposition are needed to bridge the differences between the two.
    """
    rank = len(rot.shape)
    # Swap the two last dimensions
    perm = torch.cat([torch.range(start=0, end=rank - 1), [rank], [rank - 1]], axis=0)
    return from_euler(-rot).permute(perm)


def invert_rot_and_trans(rot, trans):
    """Inverts a transform comprised of a rotation and a translation.

  Args:
    rot: a tf.Tensor of shape [..., 3] representing rotatation angles.
    trans: a tf.Tensor of shape [..., 3] representing translation vectors.

  Returns:
    a tuple (inv_rot, inv_trans), representing rotation angles and translation
    vectors, such that applting rot, transm inv_rot, inv_trans, in succession
    results in identity.
  """
    inv_rot = inverse_euler(rot)  # inv_rot = -rot  for small angles
    inv_rot_mat = matrix_from_angles(inv_rot)
    inv_trans = -torch.matmul(inv_rot_mat, torch.unsqueeze(trans, -1))
    inv_trans = torch.squeeze(inv_trans, -1)
    return inv_rot, inv_trans


def inverse_euler(angles):
    """Returns the euler angles that are the inverse of the input.

  Args:
    angles: a tf.Tensor of shape [..., 3]

  Returns:
    A tensor of the same shape, representing the inverse rotation.
  """
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    sz, sy, sx = torch.unbind(-sin_angles, axis=-1)
    cz, _, cx = torch.unbind(cos_angles, axis=-1)
    y = torch.asin((cx * sy * cz) + (sx * sz))
    x = -torch.asin((sx * sy * cz) - (cx * sz)) / torch.cos(y)
    z = -torch.asin((cx * sy * sz) - (sx * cz)) / torch.cos(y)
    return torch.stack([x, y, z], dim=-1)


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
    """Composes two transformations, each has a rotation and a translation.

  Args:
    rot_mat1: A tf.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec1: A tf.tensor of shape [..., 3] representing translation vectors.
    rot_mat2: A tf.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec2: A tf.tensor of shape [..., 3] representing translation vectors.

  Returns:
    A tuple of 2 tf.Tensors, representing rotation matrices and translation
    vectors, of the same shapes as the input, representing the result of
    applying rot1, trans1, rot2, trans2, in succession.
  """
    # Building a 4D transform matrix from each rotation and translation, and
    # multiplying the two, we'd get:
    #
    # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
    # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
    #
    # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
    # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
    # total translation is R2*t1 + t2.
    r2r1 = torch.matmul(rot_mat2, rot_mat1)
    r2t1 = torch.matmul(rot_mat2, torch.unsqueeze(trans_vec1, -1))
    r2t1 = torch.squeeze(r2t1, axis=-1)
    return r2r1, r2t1 + trans_vec2
