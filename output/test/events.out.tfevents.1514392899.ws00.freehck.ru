       �K"	  �P��Abrain.Event:2����      z㶠	���P��A"��
l
xPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
e
y_Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      F   *
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:dF*
dtype0*

seed *
T0*
seed2 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dF
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dF
�
Variable
VariableV2*
	container *&
_output_shapes
:dF*
dtype0*
shared_name *
shape:dF
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*&
_output_shapes
:dF*
T0*
_class
loc:@Variable*
use_locking(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
R
ConstConst*
dtype0*
valueBF*���=*
_output_shapes
:F
v

Variable_1
VariableV2*
	container *
_output_shapes
:F*
dtype0*
shared_name *
shape:F
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_output_shapes
:F*
T0*
_class
loc:@Variable_1*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
Conv2DConv2DReshapeVariable/read*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������dF
]
addAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������dF
K
ReluReluadd*
T0*/
_output_shapes
:���������dF
�
MaxPoolMaxPoolRelu*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������2F
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"   d   F   �   *
_output_shapes
:
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*'
_output_shapes
:dF�*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:dF�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:dF�
�

Variable_2
VariableV2*
	container *'
_output_shapes
:dF�*
dtype0*
shared_name *
shape:dF�
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*'
_output_shapes
:dF�*
T0*
_class
loc:@Variable_2*
use_locking(
x
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shared_name *
shape:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
l
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:���������2�
b
add_1AddConv2D_1Variable_3/read*
T0*0
_output_shapes
:���������2�
P
Relu_1Reluadd_1*
T0*0
_output_shapes
:���������2�
�
	MaxPool_1MaxPoolRelu_1*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:����������
i
truncated_normal_2/shapeConst*
dtype0*
valueB"�6  �  *
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
_output_shapes
:
�m�*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
�m�
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
�m�
�

Variable_4
VariableV2*
	container * 
_output_shapes
:
�m�*
dtype0*
shared_name *
shape:
�m�
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(* 
_output_shapes
:
�m�*
T0*
_class
loc:@Variable_4*
use_locking(
q
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
V
Const_2Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_5
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shared_name *
shape:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5*
use_locking(
l
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
`
Reshape_1/shapeConst*
dtype0*
valueB"�����6  *
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������m
�
MatMulMatMul	Reshape_1Variable_4/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
X
add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:����������
H
Relu_2Reluadd_2*
T0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
S
dropout/ShapeShapeRelu_2*
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*(
_output_shapes
:����������*
dtype0*

seed *
T0*
seed2 
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
dropout/divRealDivRelu_2	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	�*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shared_name *
shape:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6*
use_locking(
p
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_7
VariableV2*
	container *
_output_shapes
:*
dtype0*
shared_name *
shape:
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_7*
use_locking(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
MatMul_1MatMuldropout/mulVariable_6/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
Y
add_3AddMatMul_1Variable_7/read*
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
ShapeShapeadd_3*
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
Shape_1Shapeadd_3*
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
Slice/beginPackSub*
_output_shapes
:*
T0*
N*

axis 
T

Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
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
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*
T0*
N*

Tidx0
l
	Reshape_2Reshapeadd_3concat*
T0*
Tshape0*0
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
Slice_1/beginPackSub_1*
_output_shapes
:*
T0*
N*

axis 
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
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
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*
T0*
N*

Tidx0
k
	Reshape_3Reshapey_concat_1*
T0*
Tshape0*0
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
Slice_2/sizePackSub_2*
_output_shapes
:*
T0*
N*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_4*
	keep_dims( *
T0*

Tidx0*
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
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
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

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
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
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
T0*

Tdim0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
c
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
_output_shapes
:
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
`
gradients/dropout/div_grad/NegNegRelu_2*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:����������
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*(
_output_shapes
:����������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������m
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
�m�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:����������m
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1* 
_output_shapes
:
�m�
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:����������
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:���������2�
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:���������2�
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:���������2�
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*0
_output_shapes
:���������2�
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes	
:�
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
N*
T0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:���������2F
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:dF�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������dF
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������dF
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:F*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������dF
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:F
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*/
_output_shapes
:���������dF
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:F
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:dF
{
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape: *
shared_name *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable*
use_locking(
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape: *
shared_name *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable*
use_locking(
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueBdF*    *&
_output_shapes
:dF
�
Variable/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dF*
shared_name *&
_output_shapes
:dF
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:dF*
T0*
_class
loc:@Variable*
use_locking(
{
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueBdF*    *&
_output_shapes
:dF
�
Variable/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dF*
shared_name *&
_output_shapes
:dF
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:dF*
T0*
_class
loc:@Variable*
use_locking(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueBF*    *
_output_shapes
:F
�
Variable_1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:F*
shared_name *
_output_shapes
:F
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
T0*
_class
loc:@Variable_1*
use_locking(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueBF*    *
_output_shapes
:F
�
Variable_1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:F*
shared_name *
_output_shapes
:F
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
T0*
_class
loc:@Variable_1*
use_locking(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*&
valueBdF�*    *'
_output_shapes
:dF�
�
Variable_2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dF�*
shared_name *'
_output_shapes
:dF�
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*'
_output_shapes
:dF�*
T0*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*&
valueBdF�*    *'
_output_shapes
:dF�
�
Variable_2/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dF�*
shared_name *'
_output_shapes
:dF�
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:dF�*
T0*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueB�*    *
_output_shapes	
:�
�
Variable_3/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueB�*    *
_output_shapes	
:�
�
Variable_3/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB
�m�*    * 
_output_shapes
:
�m�
�
Variable_4/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:
�m�*
shared_name * 
_output_shapes
:
�m�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
�m�*
T0*
_class
loc:@Variable_4*
use_locking(
{
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB
�m�*    * 
_output_shapes
:
�m�
�
Variable_4/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:
�m�*
shared_name * 
_output_shapes
:
�m�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
�m�*
T0*
_class
loc:@Variable_4*
use_locking(

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5*
use_locking(
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5*
use_locking(
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
�
!Variable_6/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6*
use_locking(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6*
use_locking(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    *
_output_shapes
:
�
Variable_7/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_7*
shape:*
shared_name *
_output_shapes
:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_7*
use_locking(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    *
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_7*
shape:*
shared_name *
_output_shapes
:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_7*
use_locking(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
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
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:dF*
T0*
_class
loc:@Variable*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:F*
T0*
_class
loc:@Variable_1*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*'
_output_shapes
:dF�*
T0*
_class
loc:@Variable_2*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
�m�*
T0*
_class
loc:@Variable_4*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*
_class
loc:@Variable_5*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6*
use_nesterov( *
use_locking( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
_class
loc:@Variable_7*
use_nesterov( *
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable*
use_locking( 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
v
ArgMaxArgMaxadd_3ArgMax/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_5*
	keep_dims( *
T0*

Tidx0*
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
: "f-g@�      p��	mK�P��AJ��
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514��
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
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      F   *
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

seed *
seed2 *
T0*&
_output_shapes
:dF
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dF
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dF
�
Variable
VariableV2*
	container *
dtype0*
shape:dF*
shared_name *&
_output_shapes
:dF
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
R
ConstConst*
dtype0*
valueBF*���=*
_output_shapes
:F
v

Variable_1
VariableV2*
	container *
dtype0*
shape:F*
shared_name *
_output_shapes
:F
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
Conv2DConv2DReshapeVariable/read*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������dF
]
addAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������dF
K
ReluReluadd*
T0*/
_output_shapes
:���������dF
�
MaxPoolMaxPoolRelu*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������2F
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"   d   F   �   *
_output_shapes
:
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

seed *
seed2 *
T0*'
_output_shapes
:dF�
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:dF�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:dF�
�

Variable_2
VariableV2*
	container *
dtype0*
shape:dF�*
shared_name *'
_output_shapes
:dF�
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
x
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
	container *
dtype0*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
l
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:���������2�
b
add_1AddConv2D_1Variable_3/read*
T0*0
_output_shapes
:���������2�
P
Relu_1Reluadd_1*
T0*0
_output_shapes
:���������2�
�
	MaxPool_1MaxPoolRelu_1*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:����������
i
truncated_normal_2/shapeConst*
dtype0*
valueB"�6  �  *
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

seed *
seed2 *
T0* 
_output_shapes
:
�m�
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
�m�
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
�m�
�

Variable_4
VariableV2*
	container *
dtype0*
shape:
�m�*
shared_name * 
_output_shapes
:
�m�
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
q
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
V
Const_2Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_5
VariableV2*
	container *
dtype0*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
`
Reshape_1/shapeConst*
dtype0*
valueB"�����6  *
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������m
�
MatMulMatMul	Reshape_1Variable_4/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
X
add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:����������
H
Relu_2Reluadd_2*
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
dropout/ShapeShapeRelu_2*
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

seed *
seed2 *
T0*(
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
dropout/divRealDivRelu_2	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*

seed *
seed2 *
T0*
_output_shapes
:	�
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
	container *
dtype0*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_7
VariableV2*
	container *
dtype0*
shape:*
shared_name *
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
MatMul_1MatMuldropout/mulVariable_6/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
Y
add_3AddMatMul_1Variable_7/read*
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
ShapeShapeadd_3*
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
Shape_1Shapeadd_3*
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
T0*
N*
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
concatConcatV2concat/values_0Sliceconcat/axis*
T0*

Tidx0*
N*
_output_shapes
:
l
	Reshape_2Reshapeadd_3concat*
T0*
Tshape0*0
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
T0*
N*
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
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
T0*

Tidx0*
N*
_output_shapes
:
k
	Reshape_3Reshapey_concat_1*
T0*
Tshape0*0
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
T0*
N*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_4*
	keep_dims( *
T0*

Tidx0*
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
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
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
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
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
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
T0*

Tdim0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
c
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*'
_output_shapes
:���������
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
_output_shapes
:
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
`
gradients/dropout/div_grad/NegNegRelu_2*
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
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:����������
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*(
_output_shapes
:����������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������m
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
�m�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:����������m
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1* 
_output_shapes
:
�m�
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:����������
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*0
_output_shapes
:���������2�
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:���������2�
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:���������2�
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*0
_output_shapes
:���������2�
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes	
:�
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
N*
T0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:���������2F
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:dF�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
ksize
*
T0*
data_formatNHWC*
strides
*
paddingSAME*/
_output_shapes
:���������dF
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������dF
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:F*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������dF
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:F
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*/
_output_shapes
:���������dF
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:F
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N*
T0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:dF
{
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape: *
shared_name *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape: *
shared_name *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueBdF*    *&
_output_shapes
:dF
�
Variable/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dF*
shared_name *&
_output_shapes
:dF
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
{
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueBdF*    *&
_output_shapes
:dF
�
Variable/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dF*
shared_name *&
_output_shapes
:dF
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dF

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:dF
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueBF*    *
_output_shapes
:F
�
Variable_1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:F*
shared_name *
_output_shapes
:F
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueBF*    *
_output_shapes
:F
�
Variable_1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:F*
shared_name *
_output_shapes
:F
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:F
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*&
valueBdF�*    *'
_output_shapes
:dF�
�
Variable_2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dF�*
shared_name *'
_output_shapes
:dF�
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*&
valueBdF�*    *'
_output_shapes
:dF�
�
Variable_2/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dF�*
shared_name *'
_output_shapes
:dF�
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*'
_output_shapes
:dF�
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueB�*    *
_output_shapes	
:�
�
Variable_3/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueB�*    *
_output_shapes	
:�
�
Variable_3/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB
�m�*    * 
_output_shapes
:
�m�
�
Variable_4/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:
�m�*
shared_name * 
_output_shapes
:
�m�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
{
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB
�m�*    * 
_output_shapes
:
�m�
�
Variable_4/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:
�m�*
shared_name * 
_output_shapes
:
�m�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4* 
_output_shapes
:
�m�
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:�*
shared_name *
_output_shapes	
:�
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes	
:�
�
!Variable_6/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    *
_output_shapes
:
�
Variable_7/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_7*
shape:*
shared_name *
_output_shapes
:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    *
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_7*
shape:*
shared_name *
_output_shapes
:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
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
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *&
_output_shapes
:dF
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes
:F
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *'
_output_shapes
:dF�
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( * 
_output_shapes
:
�m�
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes	
:�
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *
_output_shapes
:	�
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
v
ArgMaxArgMaxadd_3ArgMax/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

SrcT0
*

DstT0*#
_output_shapes
:���������
Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_5*
	keep_dims( *
T0*

Tidx0*
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
: ""�
trainable_variables��
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
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0".
	summaries!

cross_entropy:0
accuracy_1:0"�
	variables��
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
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0
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
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0"
train_op

Adam�e��4       ^3\	��R��A*)

cross_entropyj�B


accuracy_1q=
?��%|6       OW��	ѣGf��A
*)

cross_entropyh� C


accuracy_1R�? ��6       OW��	���z��A*)

cross_entropy*
:B


accuracy_1   ?����6       OW��	z1����A*)

cross_entropy�� B


accuracy_1333?0�(�6       OW��	������A(*)

cross_entropy�pzB


accuracy_1�?�V�6       OW��	������A2*)

cross_entropy=�A


accuracy_1�z?�N�Y6       OW��	Z�����A<*)

cross_entropy$�JB


accuracy_1   ?7y6       OW��	���AF*)

cross_entropy���A


accuracy_1R�?؇0�6       OW��	[�����AP*)

cross_entropyx��A


accuracy_1q=
?Q��6       OW��	�����AZ*)

cross_entropyqA


accuracy_1�G?��5�6       OW��	G��"���Ad*)

cross_entropy)��A


accuracy_1q=
?�vj�6       OW��	�}�7���An*)

cross_entropy�`�A


accuracy_1��?���p6       OW��	���L���Ax*)

cross_entropy)�A


accuracy_1)\?��i�7       ���Y	���a���A�*)

cross_entropyN��A


accuracy_1
�#?2���7       ���Y	�W�v���A�*)

cross_entropy�0�A


accuracy_1��?{267       ���Y	��ɋ���A�*)

cross_entropy�cUA


accuracy_1�p=?�kh�7       ���Y	�������A�*)

cross_entropy�ܭA


accuracy_1��(?���b7       ���Y	�aҵ���A�*)

cross_entropy��A


accuracy_1333?�9��7       ���Y	W7�����A�*)

cross_entropy2�KA


accuracy_1�p=?oz�7       ���Y	JW����A�*)

cross_entropyzP�A


accuracy_1{.?^��7       ���Y	Ժ����A�*)

cross_entropy��\A


accuracy_1
�#?�#�f7       ���Y	��G
���A�*)

cross_entropy�R�@


accuracy_1{.?#H7       ���Y	w�H���A�*)

cross_entropy�B�A


accuracy_1
�#?�=��7       ���Y	��H4���A�*)

cross_entropys1�@


accuracy_1\�B?��v7       ���Y	9�RI���A�*)

cross_entropy��4A


accuracy_1R�?�5� 7       ���Y	��^���A�*)

cross_entropy#Z@


accuracy_1��Q?}f5�7       ���Y	�T�s���A�*)

cross_entropy�X1A


accuracy_1R�?'4��7       ���Y	�Xň���A�*)

cross_entropy)iXA


accuracy_1��?W�7       ���Y	�zʝ���A�*)

cross_entropy��A


accuracy_1
�#?����7       ���Y	ص�����A�*)

cross_entropyŗ�@


accuracy_1�p=?�'-�7       ���Y	|������A�*)

cross_entropy gVA


accuracy_1�z?i�{7       ���Y	>�A����A�*)

cross_entropyaKYA


accuracy_1q=
?t�a7       ���Y	]=W����A�*)

cross_entropyJ�=A


accuracy_1
�#?�L��7       ���Y	��[���A�*)

cross_entropyV��@


accuracy_1{.?��^7       ���Y	�����A�*)

cross_entropyi�>A


accuracy_1q=
?Kލ�7       ���Y	��+2���A�*)

cross_entropy��n@


accuracy_1\�B?��!%7       ���Y	�ihG���A�*)

cross_entropy��@


accuracy_1�p=?����7       ���Y	���\���A�*)

cross_entropy�gA


accuracy_1R�?�K�O7       ���Y	�#�q���A�*)

cross_entropyJt�@


accuracy_1��(?�<1�7       ���Y	蜆����A�*)

cross_entropy!-�@


accuracy_1R�?lݨ7       ���Y	�Ŝ���A�*)

cross_entropy�ʴ@


accuracy_1�Q8?0*�7       ���Y	�i����A�*)

cross_entropyqw�@


accuracy_1)\?> �;7       ���Y	G�����A�*)

cross_entropyө@


accuracy_1�Q8?���
7       ���Y	��&����A�*)

cross_entropy��@


accuracy_1{.?i��7       ���Y	�S�����A�*)

cross_entropy���@


accuracy_1�Q8?��7       ���Y	cu����A�*)

cross_entropy{�N@


accuracy_1��Q?���7       ���Y	K�1���A�*)

cross_entropyE��@


accuracy_1�Q8?H���7       ���Y	n �3���A�*)

cross_entropyw�@


accuracy_1)\?G 7k7       ���Y	j4�I���A�*)

cross_entropy�=@


accuracy_1��Q?�1�7       ���Y	OT_���A�*)

cross_entropy���@


accuracy_1��L?��z�7       ���Y	�ݼt���A�*)

cross_entropyCp�@


accuracy_1{.?"
��7       ���Y	�	p����A�*)

cross_entropy �=@


accuracy_1=
W?r�7       ���Y	�E�����A�*)

cross_entropy�H�@


accuracy_1��?Pb�7       ���Y	`Cδ���A�*)

cross_entropy3DA


accuracy_1
�#?r�;�7       ���Y	 �����A�*)

cross_entropy��@


accuracy_1{.?���7       ���Y	i�9����A�*)

cross_entropy���@


accuracy_1��(?�F�7       ���Y	r����A�*)

cross_entropyΨ�@


accuracy_1{.?�J!B7       ���Y	��	���A�*)

cross_entropy1W_@


accuracy_1��Q?����7       ���Y	D�����A�*)

cross_entropyQ��@


accuracy_1��(?��7       ���Y	�4�4���A�*)

cross_entropy�(Z@


accuracy_1�Q8?�a0s7       ���Y	��"J���A�*)

cross_entropy��O@


accuracy_1{.?)�4�7       ���Y	R�_���A�*)

cross_entropy�l@


accuracy_1333?z��7       ���Y	��Xu���A�*)

cross_entropyCG@


accuracy_1�p=?;��%7       ���Y	�qʊ���A�*)

cross_entropy��#@


accuracy_1��L?sw�7       ���Y	�.?����A�*)

cross_entropy�|�@


accuracy_1�Q8?]��7       ���Y	�b ����A�*)

cross_entropy�5�@


accuracy_1{.??���7       ���Y	!������A�*)

cross_entropy�u�?


accuracy_1=
W?C��7       ���Y	������A�*)

cross_entropy��@


accuracy_1=
W?��67       ���Y	V�C����A�*)

cross_entropy!@


accuracy_1�p=?���7       ���Y	S�����A�*)

cross_entropya�,@


accuracy_1��L?�mc7       ���Y	Ln#���A�*)

cross_entropyX�&@


accuracy_1\�B?� ��7       ���Y	�e�8���A�*)

cross_entropy\��@


accuracy_1333?��7       ���Y	o�vN���A�*)

cross_entropy�C@


accuracy_1�Q8?��n7       ���Y	���c���A�*)

cross_entropy��4@


accuracy_1��Q?һg�7       ���Y	A'qy���A�*)

cross_entropy��]@


accuracy_1333?F��7       ���Y	�	����A�*)

cross_entropyC�-@


accuracy_1�p=?�i�z7       ���Y	5�Ԥ���A�*)

cross_entropy��@


accuracy_1\�B?�)�7       ���Y	�sF����A�*)

cross_entropyO��?


accuracy_1�G?�u?7       ���Y	������A�*)

cross_entropyD�j@


accuracy_1
�#?Z�7       ���Y	��S����A�*)

cross_entropyQ@


accuracy_1�p=?�:j