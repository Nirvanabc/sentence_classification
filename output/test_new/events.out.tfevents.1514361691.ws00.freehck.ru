       �K"	  ��Ԑ�Abrain.Event:2l�3a�      !/��	���Ԑ�A"��
l
xPlaceholder*
dtype0*+
_output_shapes
:���������d* 
shape:���������d
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*
T0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      �   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*

seed *
T0*
seed2 *'
_output_shapes
:d�
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*'
_output_shapes
:d�
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*'
_output_shapes
:d�
�
Variable
VariableV2*
dtype0*'
_output_shapes
:d�*
	container *
shape:d�*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
r
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
T
ConstConst*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
Conv2DConv2DReshapeVariable/read*
paddingSAME*
strides
*
T0*0
_output_shapes
:���������d�*
use_cudnn_on_gpu(*
data_formatNHWC
^
addAddConv2DVariable_1/read*
T0*0
_output_shapes
:���������d�
L
ReluReluadd*
T0*0
_output_shapes
:���������d�
�
MaxPoolMaxPoolRelu*
paddingSAME*
ksize
*
T0*
strides
*0
_output_shapes
:���������2�*
data_formatNHWC
i
truncated_normal_1/shapeConst*
dtype0*
valueB"`�  �  *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*

seed *
T0*
seed2 *!
_output_shapes
:���
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*!
_output_shapes
:���
v
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*!
_output_shapes
:���
�

Variable_2
VariableV2*
dtype0*!
_output_shapes
:���*
	container *
shape:���*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
r
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
`
Reshape_1/shapeConst*
dtype0*
valueB"����`�  *
_output_shapes
:
p
	Reshape_1ReshapeMaxPoolReshape_1/shape*
Tshape0*
T0*)
_output_shapes
:�����������
�
MatMulMatMul	Reshape_1Variable_2/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b( 
X
add_1AddMatMulVariable_3/read*
T0*(
_output_shapes
:����������
H
Relu_1Reluadd_1*
T0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
S
dropout/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
_
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*

seed *
T0*
seed2 *(
_output_shapes
:����������
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
X
dropout/addAdd	keep_probdropout/random_uniform*
T0*
_output_shapes
:
F
dropout/FloorFloordropout/add*
T0*
_output_shapes
:
L
dropout/divRealDivRelu_1	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_2/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*

seed *
T0*
seed2 *
_output_shapes
:	�
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes
:	�
�

Variable_4
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
T
Const_2Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_5
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes
:
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:
�
MatMul_1MatMuldropout/mulVariable_4/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
Y
add_2AddMatMul_1Variable_5/read*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
J
ShapeShapeadd_2*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
L
Shape_1Shapeadd_2*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*

axis *
N*
T0*
_output_shapes
:
T

Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
b
concat/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*

Tidx0*
N*
T0*
_output_shapes
:
l
	Reshape_2Reshapeadd_2concat*
Tshape0*
T0*0
_output_shapes
:������������������
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*

axis *
N*
T0*
_output_shapes
:
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_1*
Tshape0*
T0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:
U
Slice_2/sizePackSub_2*

axis *
N*
T0*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:���������
Q
Const_3Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
cross_entropy/tagsConst*
dtype0*
valueB Bcross_entropy*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
b
gradients/add_2_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_2_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0*'
_output_shapes
:���������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	�*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*
T0*#
_output_shapes
:���������
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:���������
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0*
_output_shapes
:
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
_output_shapes
:���������
�
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
�
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:����������
`
gradients/dropout/div_grad/NegNegRelu_1*
T0*(
_output_shapes
:����������
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
�
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
�
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
�
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_1*
T0*(
_output_shapes
:����������
`
gradients/add_1_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*(
_output_shapes
:����������
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *
T0*)
_output_shapes
:�����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_1_grad/tuple/control_dependency*
transpose_a(*
T0*!
_output_shapes
:���*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:�����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:���
e
gradients/Reshape_1_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������2�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool gradients/Reshape_1_grad/Reshape*
paddingSAME*
ksize
*
T0*
strides
*0
_output_shapes
:���������d�*
data_formatNHWC
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*0
_output_shapes
:���������d�
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������d�
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*0
_output_shapes
:���������d�
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:d�
{
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_class
loc:@Variable*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
_output_shapes
: *
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
valueB
 *w�?*
_class
loc:@Variable*
_output_shapes
: 
�
beta2_power
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
_output_shapes
: *
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable*'
_output_shapes
:d�
�
Variable/Adam
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *'
_output_shapes
:d�*
	container *
shape:d�
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
|
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable*'
_output_shapes
:d�
�
Variable/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *'
_output_shapes
:d�*
	container *
shape:d�
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
�
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_1*
_output_shapes	
:�
�
Variable_1/Adam
VariableV2*
dtype0*
_class
loc:@Variable_1*
shared_name *
_output_shapes	
:�*
	container *
shape:�
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_1*
_output_shapes	
:�
�
Variable_1/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_1*
shared_name *
_output_shapes	
:�*
	container *
shape:�
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0* 
valueB���*    *
_class
loc:@Variable_2*!
_output_shapes
:���
�
Variable_2/Adam
VariableV2*
dtype0*
_class
loc:@Variable_2*
shared_name *!
_output_shapes
:���*
	container *
shape:���
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
|
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0* 
valueB���*    *
_class
loc:@Variable_2*!
_output_shapes
:���
�
Variable_2/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_2*
shared_name *!
_output_shapes
:���*
	container *
shape:���
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam
VariableV2*
dtype0*
_class
loc:@Variable_3*
shared_name *
_output_shapes	
:�*
	container *
shape:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_3*
shared_name *
_output_shapes	
:�*
	container *
shape:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_4*
_output_shapes
:	�
�
Variable_4/Adam
VariableV2*
dtype0*
_class
loc:@Variable_4*
shared_name *
_output_shapes
:	�*
	container *
shape:	�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
z
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_4*
_output_shapes
:	�
�
Variable_4/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_4*
shared_name *
_output_shapes
:	�*
	container *
shape:	�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
~
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_5*
_output_shapes
:
�
Variable_5/Adam
VariableV2*
dtype0*
_class
loc:@Variable_5*
shared_name *
_output_shapes
:*
	container *
shape:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes
:
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes
:
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_5*
_output_shapes
:
�
Variable_5/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_5*
shared_name *
_output_shapes
:*
	container *
shape:
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes
:
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable*
T0*
use_nesterov( *'
_output_shapes
:d�
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_1*
T0*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_2*
T0*
use_nesterov( *!
_output_shapes
:���
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_3*
T0*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_4*
T0*
use_nesterov( *
_output_shapes
:	�
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_5*
T0*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_class
loc:@Variable*
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
_class
loc:@Variable*
T0*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
v
ArgMaxArgMaxadd_2ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������*
output_type0	
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:���������*
output_type0	
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:���������*

SrcT0

Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
accuracy_1/tagsConst*
dtype0*
valueB B
accuracy_1*
_output_shapes
: 
W

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy*
T0*
_output_shapes
: 
^
Merge/MergeSummaryMergeSummarycross_entropy
accuracy_1*
N*
_output_shapes
: "�{,�      (��y	e���Ԑ�AJ��
�$�$
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type"
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
MaxPool

input"T
output"T"
Ttype0:
2
	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
9
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514��
l
xPlaceholder*
dtype0* 
shape:���������d*+
_output_shapes
:���������d
e
y_Placeholder*
dtype0*
shape:���������*'
_output_shapes
:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*
T0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      �   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*

seed *
T0*
seed2 *'
_output_shapes
:d�
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*'
_output_shapes
:d�
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*'
_output_shapes
:d�
�
Variable
VariableV2*
shape:d�*
dtype0*
	container *'
_output_shapes
:d�*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
_class
loc:@Variable*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:d�
r
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
T
ConstConst*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_1
VariableV2*
shape:�*
dtype0*
	container *
_output_shapes	
:�*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
_class
loc:@Variable_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
Conv2DConv2DReshapeVariable/read*
paddingSAME*
strides
*
T0*
data_formatNHWC*
use_cudnn_on_gpu(*0
_output_shapes
:���������d�
^
addAddConv2DVariable_1/read*
T0*0
_output_shapes
:���������d�
L
ReluReluadd*
T0*0
_output_shapes
:���������d�
�
MaxPoolMaxPoolRelu*
paddingSAME*
ksize
*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:���������2�
i
truncated_normal_1/shapeConst*
dtype0*
valueB"`�  �  *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*

seed *
T0*
seed2 *!
_output_shapes
:���
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*!
_output_shapes
:���
v
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*!
_output_shapes
:���
�

Variable_2
VariableV2*
shape:���*
dtype0*
	container *!
_output_shapes
:���*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
_class
loc:@Variable_2*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���
r
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
shape:�*
dtype0*
	container *
_output_shapes	
:�*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
_class
loc:@Variable_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
`
Reshape_1/shapeConst*
dtype0*
valueB"����`�  *
_output_shapes
:
p
	Reshape_1ReshapeMaxPoolReshape_1/shape*
Tshape0*
T0*)
_output_shapes
:�����������
�
MatMulMatMul	Reshape_1Variable_2/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b( 
X
add_1AddMatMulVariable_3/read*
T0*(
_output_shapes
:����������
H
Relu_1Reluadd_1*
T0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
S
dropout/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
_
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*

seed *
T0*
seed2 *(
_output_shapes
:����������
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
X
dropout/addAdd	keep_probdropout/random_uniform*
T0*
_output_shapes
:
F
dropout/FloorFloordropout/add*
T0*
_output_shapes
:
L
dropout/divRealDivRelu_1	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_2/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*

seed *
T0*
seed2 *
_output_shapes
:	�
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes
:	�
�

Variable_4
VariableV2*
shape:	�*
dtype0*
	container *
_output_shapes
:	�*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
_class
loc:@Variable_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
T
Const_2Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_5
VariableV2*
shape:*
dtype0*
	container *
_output_shapes
:*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
_class
loc:@Variable_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:
�
MatMul_1MatMuldropout/mulVariable_4/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
Y
add_2AddMatMul_1Variable_5/read*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
J
ShapeShapeadd_2*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
L
Shape_1Shapeadd_2*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*

axis *
N*
T0*
_output_shapes
:
T

Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
b
concat/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*

Tidx0*
N*
T0*
_output_shapes
:
l
	Reshape_2Reshapeadd_2concat*
Tshape0*
T0*0
_output_shapes
:������������������
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*

axis *
N*
T0*
_output_shapes
:
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_1*
Tshape0*
T0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:
U
Slice_2/sizePackSub_2*

axis *
N*
T0*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:���������
Q
Const_3Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
cross_entropy/tagsConst*
dtype0*
valueB Bcross_entropy*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
b
gradients/add_2_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_2_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0*'
_output_shapes
:���������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	�*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*
T0*#
_output_shapes
:���������
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:���������
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0*
_output_shapes
:
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
_output_shapes
:���������
�
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
�
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:����������
`
gradients/dropout/div_grad/NegNegRelu_1*
T0*(
_output_shapes
:����������
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
�
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
�
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
�
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_1*
T0*(
_output_shapes
:����������
`
gradients/add_1_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*(
_output_shapes
:����������
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *
T0*)
_output_shapes
:�����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_1_grad/tuple/control_dependency*
transpose_a(*
T0*!
_output_shapes
:���*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:�����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:���
e
gradients/Reshape_1_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������2�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool gradients/Reshape_1_grad/Reshape*
paddingSAME*
ksize
*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:���������d�
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*0
_output_shapes
:���������d�
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������d�
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*0
_output_shapes
:���������d�
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
data_formatNHWC*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
data_formatNHWC*
use_cudnn_on_gpu(*J
_output_shapes8
6:4������������������������������������
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:d�
{
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_class
loc:@Variable*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
	container *
shape: *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
loc:@Variable*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
valueB
 *w�?*
_class
loc:@Variable*
_output_shapes
: 
�
beta2_power
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
	container *
shape: *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable*'
_output_shapes
:d�
�
Variable/Adam
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
	container *
shape:d�*'
_output_shapes
:d�
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
_class
loc:@Variable*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:d�
|
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable*'
_output_shapes
:d�
�
Variable/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable*
shared_name *
	container *
shape:d�*'
_output_shapes
:d�
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
_class
loc:@Variable*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:d�
�
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*'
_output_shapes
:d�
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_1*
_output_shapes	
:�
�
Variable_1/Adam
VariableV2*
dtype0*
_class
loc:@Variable_1*
shared_name *
	container *
shape:�*
_output_shapes	
:�
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_class
loc:@Variable_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_1*
_output_shapes	
:�
�
Variable_1/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_1*
shared_name *
	container *
shape:�*
_output_shapes	
:�
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_class
loc:@Variable_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0* 
valueB���*    *
_class
loc:@Variable_2*!
_output_shapes
:���
�
Variable_2/Adam
VariableV2*
dtype0*
_class
loc:@Variable_2*
shared_name *
	container *
shape:���*!
_output_shapes
:���
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
_class
loc:@Variable_2*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���
|
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0* 
valueB���*    *
_class
loc:@Variable_2*!
_output_shapes
:���
�
Variable_2/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_2*
shared_name *
	container *
shape:���*!
_output_shapes
:���
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
_class
loc:@Variable_2*
use_locking(*
validate_shape(*
T0*!
_output_shapes
:���
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*!
_output_shapes
:���
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam
VariableV2*
dtype0*
_class
loc:@Variable_3*
shared_name *
	container *
shape:�*
_output_shapes	
:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_class
loc:@Variable_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_3*
shared_name *
	container *
shape:�*
_output_shapes	
:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_class
loc:@Variable_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:�
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_4*
_output_shapes
:	�
�
Variable_4/Adam
VariableV2*
dtype0*
_class
loc:@Variable_4*
shared_name *
	container *
shape:	�*
_output_shapes
:	�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
_class
loc:@Variable_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
z
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_4*
_output_shapes
:	�
�
Variable_4/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_4*
shared_name *
	container *
shape:	�*
_output_shapes
:	�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
_class
loc:@Variable_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	�
~
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*
_output_shapes
:	�
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_5*
_output_shapes
:
�
Variable_5/Adam
VariableV2*
dtype0*
_class
loc:@Variable_5*
shared_name *
	container *
shape:*
_output_shapes
:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_class
loc:@Variable_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes
:
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_5*
_output_shapes
:
�
Variable_5/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_5*
shared_name *
	container *
shape:*
_output_shapes
:
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_class
loc:@Variable_5*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable*
T0*
use_nesterov( *'
_output_shapes
:d�
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_1*
T0*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_2*
T0*
use_nesterov( *!
_output_shapes
:���
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_3*
T0*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_4*
T0*
use_nesterov( *
_output_shapes
:	�
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_5*
T0*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@Variable*
use_locking( *
validate_shape(*
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@Variable*
use_locking( *
validate_shape(*
T0*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
v
ArgMaxArgMaxadd_2ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������*
output_type0	
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:���������*
output_type0	
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:���������*

SrcT0

Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
accuracy_1/tagsConst*
dtype0*
valueB B
accuracy_1*
_output_shapes
: 
W

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy*
T0*
_output_shapes
: 
^
Merge/MergeSummaryMergeSummarycross_entropy
accuracy_1*
N*
_output_shapes
: ""�
	variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02Const:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0"�
trainable_variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02Const:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:0"
train_op

Adam".
	summaries!

cross_entropy:0
accuracy_1:0M��4       ^3\	):��Ԑ�A*)

cross_entropy�-�B


accuracy_1{�>s�f6       OW��	�p�Ԑ�A
*)

cross_entropy�@�A


accuracy_1�?0}ٙ6       OW��	�H��Ԑ�A*)

cross_entropyrElB


accuracy_1-�>�3�c6       OW��	$�Ԑ�A*)

cross_entropy\��A


accuracy_1D�?���E6       OW��	��G�Ԑ�A(*)

cross_entropy�BA


accuracy_1��?-��B6       OW��	�\��Ԑ�A2*)

cross_entropy}�xA


accuracy_1��?���#6       OW��	�[��Ԑ�A<*)

cross_entropyه�A


accuracy_1�?rg0\6       OW��	v�4�Ԑ�AF*)

cross_entropy��A


accuracy_1��?LЈ�6       OW��	O��Ԑ�AP*)

cross_entropy���A


accuracy_1'1?%\�6       OW��	�B��Ԑ�AZ*)

cross_entropy�XPA


accuracy_1!�?N��6       OW��	i?�Ԑ�Ad*)

cross_entropy��AA


accuracy_1!�?�N�6       OW��	;���Ԑ�An*)

cross_entropy/��@


accuracy_1�|?���v6       OW��	��-�Ԑ�Ax*)

cross_entropy/�A


accuracy_1o#?��(7       ���Y	EΝ�Ԑ�A�*)

cross_entropy	SA


accuracy_1b?��d7       ���Y	�;�Ԑ�A�*)

cross_entropy���@


accuracy_1�?f�?�7       ���Y	�Ks�Ԑ�A�*)

cross_entropy�V�@


accuracy_1m�?_��7       ���Y	J���Ԑ�A�*)

cross_entropy?Ӵ@


accuracy_1R�?;�7       ���Y	�AC Ր�A�*)

cross_entropygū@


accuracy_1/?M8�7       ���Y	(��Ր�A�*)

cross_entropy�۔@


accuracy_1��?��c�7       ���Y	uBՐ�A�*)

cross_entropy�@


accuracy_1�v?#��L7       ���Y	�Ր�A�*)

cross_entropy�͏@


accuracy_1=
?W��67       ���Y	�
Ր�A�*)

cross_entropyP�@


accuracy_1��?_�67       ���Y	���Ր�A�*)

cross_entropy��@


accuracy_1b?���7       ���Y	�?�Ր�A�*)

cross_entropym��@


accuracy_1�S?ʏ-�7       ���Y	�N[Ր�A�*)

cross_entropy�;;@


accuracy_1��?ܝ�7       ���Y	@�Ր�A�*)

cross_entropy"�6@


accuracy_1Nb?B�/7       ���Y	�e0Ր�A�*)

cross_entropyOM@


accuracy_1�z?�gI�7       ���Y	uНՐ�A�*)

cross_entropy�l@@


accuracy_1��?%�.�7       ���Y	��Ր�A�*)

cross_entropy��@


accuracy_1X9?��5�7       ���Y	�rՐ�A�*)

cross_entropy;�@


accuracy_1+�?]���7       ���Y	���Ր�A�*)

cross_entropy��@


accuracy_1��?_V�F7       ���Y	�s"Ր�A�*)

cross_entropy{��?


accuracy_1�z?"ZJ7       ���Y	�$Ր�A�*)

cross_entropyb�?


accuracy_1F�?�MyN7       ���Y	�dI'Ր�A�*)

cross_entropyw��?


accuracy_1��?�ާ7       ���Y	���)Ր�A�*)

cross_entropy�@�?


accuracy_1F�?9�C7       ���Y	T@!,Ր�A�*)

cross_entropyW��?


accuracy_1F�?�v��7       ���Y	�B�.Ր�A�*)

cross_entropy��?


accuracy_1)\?��
$7       ���Y	���0Ր�A�*)

cross_entropyik�?


accuracy_1V?At�7       ���Y	�e3Ր�A�*)

cross_entropymK�?


accuracy_1�C?��[�7       ���Y	�5Ր�A�*)

cross_entropyZQ�?


accuracy_1V?@�;�7       ���Y	X�H8Ր�A�*)

cross_entropy���?


accuracy_1{?Y\H�7       ���Y	��:Ր�A�*)

cross_entropy��?


accuracy_1��?-��7       ���Y	��M=Ր�A�*)

cross_entropy��?


accuracy_1� ?yR��7       ���Y	+�?Ր�A�*)

cross_entropy�r�?


accuracy_1V?[?T�7       ���Y	�)BՐ�A�*)

cross_entropy��?


accuracy_1��?����7       ���Y	�8�DՐ�A�*)

cross_entropy���?


accuracy_1;�?��b7       ���Y	9�GՐ�A�*)

cross_entropy�?


accuracy_1`�?%ɓ7       ���Y	�2nIՐ�A�*)

cross_entropyQ��?


accuracy_1� ?҃�J7       ���Y	�E�KՐ�A�*)

cross_entropy?


accuracy_1�&?ONA�7       ���Y	m�?NՐ�A�*)

cross_entropyd��?


accuracy_1Nb?�Ù�7       ���Y	ߨPՐ�A�*)

cross_entropyhy�?


accuracy_1�E?���7       ���Y	U�7SՐ�A�*)

cross_entropy��?


accuracy_1Nb?ʍ7       ���Y	6��UՐ�A�*)

cross_entropy��?


accuracy_1sh?;�W�7       ���Y	l�XՐ�A�*)

cross_entropy�?


accuracy_1h�?�wQ