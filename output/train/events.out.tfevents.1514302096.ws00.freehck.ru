       �K"	   ����Abrain.Event:2��k?a�      !/��	�+9����A"��
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
Reshape/shapeConst*
_output_shapes
:*%
valueB"����   d      *
dtype0
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
Tshape0*
T0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"   d      �   *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *'
_output_shapes
:d�*
seed2 *
dtype0*
T0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*'
_output_shapes
:d�*
T0
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*'
_output_shapes
:d�*
T0
�
Variable
VariableV2*'
_output_shapes
:d�*
	container *
shape:d�*
shared_name *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*'
_output_shapes
:d�*
_class
loc:@Variable*
use_locking(*
T0
r
Variable/readIdentityVariable*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
T
ConstConst*
_output_shapes	
:�*
valueB�*���=*
dtype0
x

Variable_1
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
shared_name *
dtype0
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_1*
use_locking(*
T0
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
Conv2DConv2DReshapeVariable/read*0
_output_shapes
:���������d�*
strides
*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
paddingSAME
^
addAddConv2DVariable_1/read*0
_output_shapes
:���������d�*
T0
L
ReluReluadd*0
_output_shapes
:���������d�*
T0
�
MaxPoolMaxPoolRelu*0
_output_shapes
:���������2�*
strides
*
paddingSAME*
T0*
ksize
*
data_formatNHWC
i
truncated_normal_1/shapeConst*
_output_shapes
:*
valueB"`�  �  *
dtype0
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *!
_output_shapes
:���*
seed2 *
dtype0*
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*!
_output_shapes
:���*
T0
v
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*!
_output_shapes
:���*
T0
�

Variable_2
VariableV2*!
_output_shapes
:���*
	container *
shape:���*
shared_name *
dtype0
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*!
_output_shapes
:���*
_class
loc:@Variable_2*
use_locking(*
T0
r
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
V
Const_1Const*
_output_shapes	
:�*
valueB�*���=*
dtype0
x

Variable_3
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
shared_name *
dtype0
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_3*
use_locking(*
T0
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"����`�  *
dtype0
p
	Reshape_1ReshapeMaxPoolReshape_1/shape*)
_output_shapes
:�����������*
Tshape0*
T0
�
MatMulMatMul	Reshape_1Variable_2/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
X
add_1AddMatMulVariable_3/read*(
_output_shapes
:����������*
T0
H
Relu_1Reluadd_1*(
_output_shapes
:����������*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
S
dropout/ShapeShapeRelu_1*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *(
_output_shapes
:����������*
seed2 *
dtype0*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
T0
X
dropout/addAdd	keep_probdropout/random_uniform*
_output_shapes
:*
T0
F
dropout/FloorFloordropout/add*
_output_shapes
:*
T0
L
dropout/divRealDivRelu_1	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB"�     *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
_output_shapes
:	�*
seed2 *
dtype0*
T0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
_output_shapes
:	�*
T0
�

Variable_4
VariableV2*
_output_shapes
:	�*
	container *
shape:	�*
shared_name *
dtype0
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_output_shapes
:	�*
_class
loc:@Variable_4*
use_locking(*
T0
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
T
Const_2Const*
_output_shapes
:*
valueB*���=*
dtype0
v

Variable_5
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_output_shapes
:*
_class
loc:@Variable_5*
use_locking(*
T0
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
MatMul_1MatMuldropout/mulVariable_4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
add_2AddMatMul_1Variable_5/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
J
ShapeShapeadd_2*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
L
Shape_1Shapeadd_2*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*
N*

axis *
T0
T

Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*
N*

Tidx0*
T0
l
	Reshape_2Reshapeadd_2concat*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*
N*

axis *
T0
V
Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*
N*

Tidx0*
T0
k
	Reshape_3Reshapey_concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:���������:������������������*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
U
Slice_2/sizePackSub_2*
_output_shapes
:*
N*

axis *
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
Tshape0*
T0
Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0
^
MeanMean	Reshape_4Const_3*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
_output_shapes
: *
valueB Bcross_entropy*
dtype0
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
b
gradients/add_2_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:���������*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:����������*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes
:	�*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*#
_output_shapes
:���������*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
f
 gradients/dropout/div_grad/ShapeShapeRelu_1*
out_type0*
_output_shapes
:*
T0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*#
_output_shapes
:���������*
T0
�
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
_output_shapes
:*
T0
�
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
`
gradients/dropout/div_grad/NegNegRelu_1*(
_output_shapes
:����������*
T0
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0
�
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
_output_shapes
:*
T0
�
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������*
T0
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_1*(
_output_shapes
:����������*
T0
`
gradients/add_1_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*)
_output_shapes
:�����������*
transpose_a( *
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_1_grad/tuple/control_dependency*!
_output_shapes
:���*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*)
_output_shapes
:�����������*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*!
_output_shapes
:���*
T0
e
gradients/Reshape_1_grad/ShapeShapeMaxPool*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:���������2�*
Tshape0*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool gradients/Reshape_1_grad/Reshape*0
_output_shapes
:���������d�*
strides
*
paddingSAME*
T0*
ksize
*
data_formatNHWC
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*0
_output_shapes
:���������d�*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*0
_output_shapes
:���������d�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*0
_output_shapes
:���������d�*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes	
:�*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N* 
_output_shapes
::*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
strides
*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
paddingSAME
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
strides
*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
paddingSAME
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:d�*
T0
{
beta1_power/initial_valueConst*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *fff?*
dtype0
�
beta1_power
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
_class
loc:@Variable*
use_locking(*
T0
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
_output_shapes
: *
T0
{
beta2_power/initial_valueConst*
_output_shapes
: *
_class
loc:@Variable*
valueB
 *w�?*
dtype0
�
beta2_power
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
_class
loc:@Variable*
use_locking(*
T0
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Variable/Adam/Initializer/zerosConst*'
_output_shapes
:d�*
_class
loc:@Variable*&
valueBd�*    *
dtype0
�
Variable/Adam
VariableV2*'
_output_shapes
:d�*
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape:d�
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*'
_output_shapes
:d�*
_class
loc:@Variable*
use_locking(*
T0
|
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
�
!Variable/Adam_1/Initializer/zerosConst*'
_output_shapes
:d�*
_class
loc:@Variable*&
valueBd�*    *
dtype0
�
Variable/Adam_1
VariableV2*'
_output_shapes
:d�*
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape:d�
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:d�*
_class
loc:@Variable*
use_locking(*
T0
�
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes	
:�*
_class
loc:@Variable_1*
valueB�*    *
dtype0
�
Variable_1/Adam
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_1*
shape:�
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_1*
use_locking(*
T0
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
_class
loc:@Variable_1*
valueB�*    *
dtype0
�
Variable_1/Adam_1
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_1*
shape:�
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_1*
use_locking(*
T0
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
!Variable_2/Adam/Initializer/zerosConst*!
_output_shapes
:���*
_class
loc:@Variable_2* 
valueB���*    *
dtype0
�
Variable_2/Adam
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
_class
loc:@Variable_2*
shape:���
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*!
_output_shapes
:���*
_class
loc:@Variable_2*
use_locking(*
T0
|
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*!
_output_shapes
:���*
_class
loc:@Variable_2* 
valueB���*    *
dtype0
�
Variable_2/Adam_1
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
_class
loc:@Variable_2*
shape:���
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*!
_output_shapes
:���*
_class
loc:@Variable_2*
use_locking(*
T0
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
_output_shapes	
:�*
_class
loc:@Variable_3*
valueB�*    *
dtype0
�
Variable_3/Adam
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_3*
shape:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_3*
use_locking(*
T0
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
_class
loc:@Variable_3*
valueB�*    *
dtype0
�
Variable_3/Adam_1
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_3*
shape:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
_class
loc:@Variable_3*
use_locking(*
T0
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
_output_shapes
:	�*
_class
loc:@Variable_4*
valueB	�*    *
dtype0
�
Variable_4/Adam
VariableV2*
_output_shapes
:	�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_4*
shape:	�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
_class
loc:@Variable_4*
use_locking(*
T0
z
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
_class
loc:@Variable_4*
valueB	�*    *
dtype0
�
Variable_4/Adam_1
VariableV2*
_output_shapes
:	�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_4*
shape:	�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
_class
loc:@Variable_4*
use_locking(*
T0
~
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
�
!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@Variable_5*
valueB*    *
dtype0
�
Variable_5/Adam
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
_class
loc:@Variable_5*
use_locking(*
T0
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
#Variable_5/Adam_1/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@Variable_5*
valueB*    *
dtype0
�
Variable_5/Adam_1
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
_class
loc:@Variable_5*
use_locking(*
T0
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *��8*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*'
_output_shapes
:d�*
use_nesterov( *
_class
loc:@Variable*
use_locking( *
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_nesterov( *
_class
loc:@Variable_1*
use_locking( *
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*!
_output_shapes
:���*
use_nesterov( *
_class
loc:@Variable_2*
use_locking( *
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_nesterov( *
_class
loc:@Variable_3*
use_locking( *
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes
:	�*
use_nesterov( *
_class
loc:@Variable_4*
use_locking( *
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *
_class
loc:@Variable_5*
use_locking( *
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
_class
loc:@Variable*
use_locking( *
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
_class
loc:@Variable*
use_locking( *
T0
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
v
ArgMaxArgMaxadd_2ArgMax/dimension*

Tidx0*#
_output_shapes
:���������*
output_type0	*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*#
_output_shapes
:���������*
output_type0	*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*

SrcT0
*

DstT0*#
_output_shapes
:���������
Q
Const_4Const*
_output_shapes
:*
valueB: *
dtype0
_
accuracyMeanCast_1Const_4*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
_output_shapes
: *
valueB B
accuracy_1*
dtype0
W

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy*
_output_shapes
: *
T0
^
Merge/MergeSummaryMergeSummarycross_entropy
accuracy_1*
N*
_output_shapes
: "`� �,�      (��y	PDA����AJ��
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
Reshape/shapeConst*
_output_shapes
:*%
valueB"����   d      *
dtype0
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
Tshape0*
T0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"   d      �   *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *'
_output_shapes
:d�*
T0*
seed2 *
dtype0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*'
_output_shapes
:d�*
T0
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*'
_output_shapes
:d�*
T0
�
Variable
VariableV2*'
_output_shapes
:d�*
	container *
shape:d�*
shared_name *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*'
_output_shapes
:d�*
T0*
_class
loc:@Variable*
use_locking(
r
Variable/readIdentityVariable*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
T
ConstConst*
_output_shapes	
:�*
valueB�*���=*
dtype0
x

Variable_1
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
shared_name *
dtype0
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_1*
use_locking(
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
Conv2DConv2DReshapeVariable/read*0
_output_shapes
:���������d�*
strides
*
use_cudnn_on_gpu(*
T0*
paddingSAME*
data_formatNHWC
^
addAddConv2DVariable_1/read*0
_output_shapes
:���������d�*
T0
L
ReluReluadd*0
_output_shapes
:���������d�*
T0
�
MaxPoolMaxPoolRelu*0
_output_shapes
:���������2�*
strides
*
paddingSAME*
T0*
ksize
*
data_formatNHWC
i
truncated_normal_1/shapeConst*
_output_shapes
:*
valueB"`�  �  *
dtype0
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *!
_output_shapes
:���*
T0*
seed2 *
dtype0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*!
_output_shapes
:���*
T0
v
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*!
_output_shapes
:���*
T0
�

Variable_2
VariableV2*!
_output_shapes
:���*
	container *
shape:���*
shared_name *
dtype0
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*!
_output_shapes
:���*
T0*
_class
loc:@Variable_2*
use_locking(
r
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
V
Const_1Const*
_output_shapes	
:�*
valueB�*���=*
dtype0
x

Variable_3
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
shared_name *
dtype0
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"����`�  *
dtype0
p
	Reshape_1ReshapeMaxPoolReshape_1/shape*)
_output_shapes
:�����������*
Tshape0*
T0
�
MatMulMatMul	Reshape_1Variable_2/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
X
add_1AddMatMulVariable_3/read*(
_output_shapes
:����������*
T0
H
Relu_1Reluadd_1*(
_output_shapes
:����������*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
S
dropout/ShapeShapeRelu_1*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *(
_output_shapes
:����������*
T0*
seed2 *
dtype0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
T0
X
dropout/addAdd	keep_probdropout/random_uniform*
_output_shapes
:*
T0
F
dropout/FloorFloordropout/add*
_output_shapes
:*
T0
L
dropout/divRealDivRelu_1	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB"�     *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
_output_shapes
:	�*
T0*
seed2 *
dtype0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
_output_shapes
:	�*
T0
�

Variable_4
VariableV2*
_output_shapes
:	�*
	container *
shape:	�*
shared_name *
dtype0
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_4*
use_locking(
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
T
Const_2Const*
_output_shapes
:*
valueB*���=*
dtype0
v

Variable_5
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_5*
use_locking(
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
MatMul_1MatMuldropout/mulVariable_4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
add_2AddMatMul_1Variable_5/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
J
ShapeShapeadd_2*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
L
Shape_1Shapeadd_2*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*
T0*
N*

axis 
T

Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
Index0*
T0
b
concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*

Tidx0*
N*
T0
l
	Reshape_2Reshapeadd_2concat*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*
T0*
N*

axis 
V
Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
Index0*
T0
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*

Tidx0*
N*
T0
k
	Reshape_3Reshapey_concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:���������:������������������*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
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
:���������*
Index0*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
Tshape0*
T0
Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0
^
MeanMean	Reshape_4Const_3*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
_output_shapes
: *
valueB Bcross_entropy*
dtype0
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
b
gradients/add_2_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*'
_output_shapes
:���������*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*(
_output_shapes
:����������*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes
:	�*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*#
_output_shapes
:���������*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
_output_shapes
:*
T0
f
 gradients/dropout/div_grad/ShapeShapeRelu_1*
out_type0*
_output_shapes
:*
T0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*#
_output_shapes
:���������*
T0
�
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
_output_shapes
:*
T0
�
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
`
gradients/dropout/div_grad/NegNegRelu_1*(
_output_shapes
:����������*
T0
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0
�
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
_output_shapes
:*
T0
�
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������*
T0
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_1*(
_output_shapes
:����������*
T0
`
gradients/add_1_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_2/read*)
_output_shapes
:�����������*
transpose_a( *
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_1_grad/tuple/control_dependency*!
_output_shapes
:���*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*)
_output_shapes
:�����������*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*!
_output_shapes
:���*
T0
e
gradients/Reshape_1_grad/ShapeShapeMaxPool*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:���������2�*
Tshape0*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool gradients/Reshape_1_grad/Reshape*0
_output_shapes
:���������d�*
strides
*
paddingSAME*
T0*
ksize
*
data_formatNHWC
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*0
_output_shapes
:���������d�*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*0
_output_shapes
:���������d�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*0
_output_shapes
:���������d�*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes	
:�*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N* 
_output_shapes
::*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
strides
*
use_cudnn_on_gpu(*
T0*
paddingSAME*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
strides
*
use_cudnn_on_gpu(*
T0*
paddingSAME*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:d�*
T0
{
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable*
dtype0
�
beta1_power
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape: 
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
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
_output_shapes
: *
T0
{
beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@Variable*
dtype0
�
beta2_power
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape: 
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
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Variable/Adam/Initializer/zerosConst*'
_output_shapes
:d�*&
valueBd�*    *
_class
loc:@Variable*
dtype0
�
Variable/Adam
VariableV2*'
_output_shapes
:d�*
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape:d�
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*'
_output_shapes
:d�*
T0*
_class
loc:@Variable*
use_locking(
|
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
�
!Variable/Adam_1/Initializer/zerosConst*'
_output_shapes
:d�*&
valueBd�*    *
_class
loc:@Variable*
dtype0
�
Variable/Adam_1
VariableV2*'
_output_shapes
:d�*
	container *
dtype0*
shared_name *
_class
loc:@Variable*
shape:d�
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:d�*
T0*
_class
loc:@Variable*
use_locking(
�
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*'
_output_shapes
:d�*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Adam
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_1*
shape:�
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_1*
use_locking(
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Adam_1
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_1*
shape:�
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_1*
use_locking(
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
_output_shapes	
:�*
T0
�
!Variable_2/Adam/Initializer/zerosConst*!
_output_shapes
:���* 
valueB���*    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
_class
loc:@Variable_2*
shape:���
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*!
_output_shapes
:���*
T0*
_class
loc:@Variable_2*
use_locking(
|
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*!
_output_shapes
:���* 
valueB���*    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam_1
VariableV2*!
_output_shapes
:���*
	container *
dtype0*
shared_name *
_class
loc:@Variable_2*
shape:���
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*!
_output_shapes
:���*
T0*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*!
_output_shapes
:���*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3*
dtype0
�
Variable_3/Adam
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_3*
shape:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3*
dtype0
�
Variable_3/Adam_1
VariableV2*
_output_shapes	
:�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_3*
shape:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
_output_shapes	
:�*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam
VariableV2*
_output_shapes
:	�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_4*
shape:	�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_4*
use_locking(
z
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam_1
VariableV2*
_output_shapes
:	�*
	container *
dtype0*
shared_name *
_class
loc:@Variable_4*
shape:	�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
T0*
_class
loc:@Variable_4*
use_locking(
~
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
_output_shapes
:	�*
T0
�
!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_5*
dtype0
�
Variable_5/Adam
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_5*
use_locking(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
#Variable_5/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_5*
dtype0
�
Variable_5/Adam_1
VariableV2*
_output_shapes
:*
	container *
dtype0*
shared_name *
_class
loc:@Variable_5*
shape:
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
_class
loc:@Variable_5*
use_locking(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *��8*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *'
_output_shapes
:d�*
T0*
_class
loc:@Variable*
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
T0*
_class
loc:@Variable_1*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *!
_output_shapes
:���*
T0*
_class
loc:@Variable_2*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
T0*
_class
loc:@Variable_3*
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	�*
T0*
_class
loc:@Variable_4*
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
T0*
_class
loc:@Variable_5*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0
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
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
v
ArgMaxArgMaxadd_2ArgMax/dimension*

Tidx0*#
_output_shapes
:���������*
output_type0	*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*#
_output_shapes
:���������*
output_type0	*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*

SrcT0
*

DstT0*#
_output_shapes
:���������
Q
Const_4Const*
_output_shapes
:*
valueB: *
dtype0
_
accuracyMeanCast_1Const_4*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
_output_shapes
: *
valueB B
accuracy_1*
dtype0
W

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy*
_output_shapes
: *
T0
^
Merge/MergeSummaryMergeSummarycross_entropy
accuracy_1*
N*
_output_shapes
: "".
	summaries!

cross_entropy:0
accuracy_1:0"
train_op

Adam"�
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
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:0"�
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
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0b�:4       ^3\	��N����A*)

cross_entropy�� B


accuracy_1   ?�Z6       OW��	Q�u����A*)

cross_entropy�:C


accuracy_1)\?�f��6       OW��	~t�����A*)

cross_entropy���B


accuracy_1�Q�>�b�6       OW��	��Ħ���A*)

cross_entropy?�B


accuracy_1���>���6       OW��	��즚��A*)

cross_entropysOmB


accuracy_1��?[�N�6       OW��	�����A*)

cross_entropy���B


accuracy_1R�?�V�6       OW��	��;����A*)

cross_entropys�B


accuracy_1)\?����6       OW��	�Rc����A*)

cross_entropy�TB


accuracy_1R�?B�o6       OW��	������A*)

cross_entropyl6�A


accuracy_1�?S�h_6       OW��	�̳����A	*)

cross_entropy	ZCB


accuracy_1)\?����6       OW��	��V����A
*)

cross_entropy}B


accuracy_1R�?�_�M6       OW��	3�}����A*)

cross_entropy�y�B


accuracy_1�G�>V�6       OW��	������A*)

cross_entropypY�B


accuracy_1q=
?�cl�6       OW��	LΩ���A*)

cross_entropy��B


accuracy_1���>D�C6       OW��	4�����A*)

cross_entropym��A


accuracy_1
�#?���6       OW��	�����A*)

cross_entropy�:B


accuracy_1q=
?��r!6       OW��	�"J����A*)

cross_entropyxP�A


accuracy_1)\?}��q6       OW��	eq����A*)

cross_entropy6H B


accuracy_1�z?KaU6       OW��	�-�����A*)

cross_entropy·B


accuracy_1�?k��6       OW��	P�����A*)

cross_entropy���A


accuracy_1�?�]��6       OW��	�_����A*)

cross_entropyJ�A


accuracy_1q=
?���6       OW��	X�����A*)

cross_entropy/��A


accuracy_1q=
?�O6       OW��	�7�����A*)

cross_entropy��A


accuracy_1   ?�[$f6       OW��	�k׬���A*)

cross_entropyφ�A


accuracy_1��?��<W6       OW��	%������A*)

cross_entropy�p$B


accuracy_1�?O:��6       OW��	�8'����A*)

cross_entropyV�A


accuracy_1��(?Ɓ[F6       OW��	.O����A*)

cross_entropyæB


accuracy_1�?��6       OW��	��v����A*)

cross_entropy�_B


accuracy_1��?��DJ6       OW��	������A*)

cross_entropyr�AB


accuracy_1��>=U6       OW��	&�ƭ���A*)

cross_entropyo�A


accuracy_1q=
?q��D6       OW��	)�h����A*)

cross_entropyҦA


accuracy_1q=
? �<�6       OW��	�b�����A*)

cross_entropyV�A


accuracy_1�G�>H�Ț6       OW��	ڸ����A *)

cross_entropyǵA


accuracy_1)\?��8�6       OW��	�H௚��A!*)

cross_entropyR�A


accuracy_1R�?Ig�6       OW��	l�	����A"*)

cross_entropy���A


accuracy_1
�#?�h��6       OW��	�l0����A#*)

cross_entropy[��A


accuracy_1�?���6       OW��	?]Y����A$*)

cross_entropy&��A


accuracy_1   ?��ߘ6       OW��	타����A%*)

cross_entropy�h B


accuracy_1=
�>=2�36       OW��	�2�����A&*)

cross_entropy	�^A


accuracy_1R�?��8�6       OW��	}Ѱ���A'*)

cross_entropy�:�A


accuracy_1q=
?|U>�6       OW��	HNx����A(*)

cross_entropy`l�A


accuracy_1�z?vU��6       OW��	�j�����A)*)

cross_entropy��B


accuracy_1R�?�u 6       OW��	bȲ���A**)

cross_entropyz�$B


accuracy_1
�#?����6       OW��	�A󲚐�A+*)

cross_entropy1�A


accuracy_1q=
?��6       OW��	������A,*)

cross_entropy��A


accuracy_1�z?��C6       OW��	��C����A-*)

cross_entropyU��A


accuracy_1)\?�)�6       OW��	Pk����A.*)

cross_entropy�@�A


accuracy_1)\?`��&6       OW��	�W�����A/*)

cross_entropyT�A


accuracy_1�?��86       OW��	������A0*)

cross_entropy6��A


accuracy_1��?ȵ��6       OW��	��峚��A1*)

cross_entropy_�eA


accuracy_1�Q8?�G��6       OW��	r�����A2*)

cross_entropy��A


accuracy_1�z?j�6       OW��	�ŵ���A3*)

cross_entropy���A


accuracy_1���>�6�6       OW��	�������A4*)

cross_entropy_��A


accuracy_1�z?���`6       OW��	������A5*)

cross_entropy�	aA


accuracy_1�z?ڈ
�6       OW��	�_=����A6*)

cross_entropy�ټA


accuracy_1��>k̫.6       OW��	��e����A7*)

cross_entropy�M�A


accuracy_1)\?"��6       OW��	�k�����A8*)

cross_entropy��WA


accuracy_1R�?OЩG6       OW��	"춶���A9*)

cross_entropy;��A


accuracy_1q=
?J�a6       OW��	�	ක��A:*)

cross_entropyN��A


accuracy_1
�#?�;K6       OW��	�R����A;*)

cross_entropy�8�A


accuracy_1q=
?����6       OW��	M𾸚��A<*)

cross_entropy�M�A


accuracy_1�z?���c6       OW��	e4渚��A=*)

cross_entropy֡�A


accuracy_1�?ƨ�6       OW��	�\����A>*)

cross_entropy&�A


accuracy_1)\?���6       OW��	tY6����A?*)

cross_entropyA1A


accuracy_1
�#?C��x6       OW��	E,^����A@*)

cross_entropy�E�A


accuracy_1�z?��
�6       OW��	[ą����AA*)

cross_entropy�-�A


accuracy_1��(?Y��R6       OW��	�ɮ����AB*)

cross_entropy���A


accuracy_1   ?\�6       OW��	��׹���AC*)

cross_entropy��lA


accuracy_1
�#?�@W
6       OW��	������AD*)

cross_entropy�(gA


accuracy_1{.?+G��6       OW��	bt+����AE*)

cross_entropy�ǙA


accuracy_1��?��6       OW��	��绚��AF*)

cross_entropy18�A


accuracy_1R�?�0M�6       OW��	N=����AG*)

cross_entropy�2�A


accuracy_1
�#?t`�L6       OW��	�S9����AH*)

cross_entropy�z�A


accuracy_1���>:$ 46       OW��	Q~a����AI*)

cross_entropy��	B


accuracy_1���>�K��6       OW��	P4�����AJ*)

cross_entropyI�xA


accuracy_1   ?o��6       OW��	�@�����AK*)

cross_entropy���A


accuracy_1q=
?.;kJ6       OW��	�ۼ���AL*)

cross_entropyPgB


accuracy_1   ?�l56       OW��	�����AM*)

cross_entropy�4�A


accuracy_1R�?G���6       OW��	��-����AN*)

cross_entropy��iA


accuracy_1��?[I��6       OW��	�!X����AO*)

cross_entropyo�MA


accuracy_1333?"���6       OW��	�y����AP*)

cross_entropy���A


accuracy_1��>���6       OW��	[f=����AQ*)

cross_entropy{��A


accuracy_1=
�>��76       OW��	xhf����AR*)

cross_entropy��TA


accuracy_1
�#?<�06       OW��	6�����AS*)

cross_entropyb/�A


accuracy_1�?�0�!6       OW��	�ڶ����AT*)

cross_entropy0��A


accuracy_1R�?HN6       OW��	"�߿���AU*)

cross_entropy�+�A


accuracy_1q=
?u׹z6       OW��	lK	����AV*)

cross_entropyJ��A


accuracy_1)\?�K��6       OW��	�63����AW*)

cross_entropy�,iA


accuracy_1�z?$'�6       OW��	wK\����AX*)

cross_entropy�mA


accuracy_1��?�9�6       OW��	e؇����AY*)

cross_entropy�>mA


accuracy_1)\?�>�D6       OW��	��O��AZ*)

cross_entropy���A


accuracy_1q=
?���6       OW��	I�x��A[*)

cross_entropy��\A


accuracy_1)\?�R6       OW��	ߧ���A\*)

cross_entropy��qA


accuracy_1�z?��^46       OW��	�����A]*)

cross_entropyc��A


accuracy_1���>P_>6       OW��	O����A^*)

cross_entropy(�A


accuracy_1��?Qb�\6       OW��	��Ú��A_*)

cross_entropydJA


accuracy_1)\?8P�6       OW��	�xGÚ��A`*)

cross_entropy��AA


accuracy_1
�#?�㿢6       OW��	�qÚ��Aa*)

cross_entropy�VZA


accuracy_1
�#?Y���6       OW��	?ÚÚ��Ab*)

cross_entropy�h�A


accuracy_1)\?�n6       OW��	K��Ú��Ac*)

cross_entropy�(A


accuracy_1�z?&�[�6       OW��	ǎ�Ś��Ad*)

cross_entropy�SFA


accuracy_1��(?��w�6       OW��	�Ś��Ae*)

cross_entropy�A


accuracy_1   ?����6       OW��	Y��Ś��Af*)

cross_entropy]AA


accuracy_1q=
?Q��6       OW��	(�Ś��Ag*)

cross_entropy���A


accuracy_1)\?�z�6       OW��	�(ƚ��Ah*)

cross_entropyR�KA


accuracy_1)\?뻷�6       OW��	�<Rƚ��Ai*)

cross_entropy�DeA


accuracy_1)\?Lӈ6       OW��	#�{ƚ��Aj*)

cross_entropy��A


accuracy_1)\?�R��6       OW��	7a�ƚ��Ak*)

cross_entropy8%A


accuracy_1
�#?^��6       OW��	
q�ƚ��Al*)

cross_entropy��A


accuracy_1���>�I�6       OW��	w��ƚ��Am*)

cross_entropy�KDA


accuracy_1�G�>�k`6       OW��	���Ț��An*)

cross_entropy�A


accuracy_1R�?�\�56       OW��	nF�Ț��Ao*)

cross_entropy�A


accuracy_1�z??�"6       OW��	thɚ��Ap*)

cross_entropy�O�@


accuracy_1��(?T�gd6       OW��	�I:ɚ��Aq*)

cross_entropy�NkA


accuracy_1���>pH4�6       OW��	G�cɚ��Ar*)

cross_entropy�6�@


accuracy_1333?�62�6       OW��	X��ɚ��As*)

cross_entropy�B


accuracy_1��?��ɷ6       OW��	���ɚ��At*)

cross_entropy�/A


accuracy_1��?�l��6       OW��	��ɚ��Au*)

cross_entropy��A


accuracy_1��(?�!9�6       OW��	G0ʚ��Av*)

cross_entropy��/A


accuracy_1��(?�Ї�6       OW��	��6ʚ��Aw*)

cross_entropy(�@


accuracy_1�z?m�6       OW��	���˚��Ax*)

cross_entropyjFBA


accuracy_1q=
?�#S46       OW��	=�&̚��Ay*)

cross_entropy�\A


accuracy_1���>d��.6       OW��	b�O̚��Az*)

cross_entropy��1A


accuracy_1��(?S��S6       OW��	�x̚��A{*)

cross_entropym�YA


accuracy_1�?HQVr6       OW��	&E�̚��A|*)

cross_entropyjŖ@


accuracy_1�Q8?C%�6       OW��	�L�̚��A}*)

cross_entropyS`CA


accuracy_1�?e�j)6       OW��	+��̚��A~*)

cross_entropy�jcA


accuracy_1   ?]d=�6       OW��	�� ͚��A*)

cross_entropy�%A


accuracy_1�z?X+R�7       ���Y	�wJ͚��A�*)

cross_entropya�2A


accuracy_1�?��Y7       ���Y	3|v͚��A�*)

cross_entropy�i'A


accuracy_1R�?�}�7       ���Y	�BϚ��A�*)

cross_entropy�UBA


accuracy_1�z?����7       ���Y	�@lϚ��A�*)

cross_entropyQc	A


accuracy_1)\??Hx�7       ���Y	+�Ϛ��A�*)

cross_entropy,0$A


accuracy_1)\?��z7       ���Y	�ԾϚ��A�*)

cross_entropy�kA


accuracy_1q=
?=�!7       ���Y	NS�Ϛ��A�*)

cross_entropy��*A


accuracy_1�?1}6N7       ���Y	rFК��A�*)

cross_entropymуA


accuracy_1   ?�i!}7       ���Y	΂=К��A�*)

cross_entropy+�A


accuracy_1R�?��7       ���Y	�fК��A�*)

cross_entropy�d)A


accuracy_1��>����7       ���Y	���К��A�*)

cross_entropy��A


accuracy_1\��>��R7       ���Y	�К��A�*)

cross_entropy��$A


accuracy_1��?���7       ���Y	��}Қ��A�*)

cross_entropy��@


accuracy_1333?:3Z�7       ���Y	�)�Қ��A�*)

cross_entropy o�@


accuracy_1��?��7       ���Y	��Қ��A�*)

cross_entropy��A


accuracy_1)\?��7       ���Y	#��Қ��A�*)

cross_entropyr��@


accuracy_1��(?F�\7       ���Y	#u#Ӛ��A�*)

cross_entropy�j�A


accuracy_1���>]��n7       ���Y	2NӚ��A�*)

cross_entropy�HA


accuracy_1���>3[�7       ���Y	0zӚ��A�*)

cross_entropyJ��A


accuracy_1��>PMjT7       ���Y	��Ӛ��A�*)

cross_entropyO.A


accuracy_1)\?��7       ���Y	p,�Ӛ��A�*)

cross_entropy�N8A


accuracy_1   ?�7�S7       ���Y	���Ӛ��A�*)

cross_entropy���@


accuracy_1R�?�0"�7       ���Y	+�՚��A�*)

cross_entropy���@


accuracy_1��(?A7       ���Y	���՚��A�*)

cross_entropy�,�@


accuracy_1R�?�-q�7       ���Y	�G֚��A�*)

cross_entropy�N�@


accuracy_1�z?h�()7       ���Y	Q�@֚��A�*)

cross_entropyq�6A


accuracy_1)\?=s!v7       ���Y	��l֚��A�*)

cross_entropy	�@


accuracy_1R�?�(�7       ���Y	��֚��A�*)

cross_entropy!��@


accuracy_1)\?%R��7       ���Y	��֚��A�*)

cross_entropym/~A


accuracy_1���>�n��7       ���Y	�7�֚��A�*)

cross_entropy �A


accuracy_1��?*P�L7       ���Y	�Yך��A�*)

cross_entropy{.�@


accuracy_1�?2�$�7       ���Y	�;ך��A�*)

cross_entropy_��@


accuracy_1�?�-��7       ���Y		�ٚ��A�*)

cross_entropy��A


accuracy_1q=
?�N�7       ���Y	��0ٚ��A�*)

cross_entropy{LA


accuracy_1333?���7       ���Y	<[ٚ��A�*)

cross_entropy��/A


accuracy_1q=
?M���7       ���Y	���ٚ��A�*)

cross_entropy7��@


accuracy_1{.?�Б*7       ���Y	��ٚ��A�*)

cross_entropy/t�@


accuracy_1333?&	{�7       ���Y	d�ٚ��A�*)

cross_entropy��@


accuracy_1R�?!҉T7       ���Y	�Uښ��A�*)

cross_entropyj�A


accuracy_1���>�!z�7       ���Y	F�,ښ��A�*)

cross_entropy�?A


accuracy_1�z?e�7       ���Y	�Vښ��A�*)

cross_entropy|��@


accuracy_1q=
?�F�7       ���Y	���ښ��A�*)

cross_entropy=zA


accuracy_1R�?�Z�7       ���Y	o3?ܚ��A�*)

cross_entropy�:�@


accuracy_1   ?�gp�7       ���Y	�+iܚ��A�*)

cross_entropy�@


accuracy_1)\?ȁA�7       ���Y	�ܚ��A�*)

cross_entropypA


accuracy_1�?���.7       ���Y	��ܚ��A�*)

cross_entropy4A


accuracy_1   ?]���7       ���Y	��ܚ��A�*)

cross_entropyVB A


accuracy_1)\?����7       ���Y	�Tݚ��A�*)

cross_entropy���@


accuracy_1�z?"�7       ���Y	U:ݚ��A�*)

cross_entropy�r�@


accuracy_1q=
?j�'H7       ���Y	�Sdݚ��A�*)

cross_entropy�1�@


accuracy_1)\?}q2�7       ���Y	�ݚ��A�*)

cross_entropy#��@


accuracy_1   ?��a�7       ���Y	�ݚ��A�*)

cross_entropy긜@


accuracy_1�z?���7       ���Y	��ߚ��A�*)

cross_entropy Gc@


accuracy_1
�#?g��V7       ���Y	M�ߚ��A�*)

cross_entropy� �@


accuracy_1�z?��S\7       ���Y	:��ߚ��A�*)

cross_entropy�Ϲ@


accuracy_1q=
?�J�q7       ���Y	�R����A�*)

cross_entropy�6�?


accuracy_1�p=?7YG�7       ���Y	��+����A�*)

cross_entropy=��@


accuracy_1�z?�5�i7       ���Y	P�U����A�*)

cross_entropy���@


accuracy_1�p=?
v^�7       ���Y	&�~����A�*)

cross_entropy @@


accuracy_1{.?_df7       ���Y	@������A�*)

cross_entropy��@


accuracy_1R�?���j7       ���Y	2`�����A�*)

cross_entropy��@


accuracy_1�?H�-7       ���Y	׿ ᚐ�A�*)

cross_entropyOڛ@


accuracy_1   ?��G�7       ���Y	�(�⚐�A�*)

cross_entropy1i�@


accuracy_1��?���~7       ���Y	�t�⚐�A�*)

cross_entropyZ`O@


accuracy_1q=
?ȼ��7       ���Y	=�#㚐�A�*)

cross_entropy�s�@


accuracy_1q=
?B͆7       ���Y	�M㚐�A�*)

cross_entropy�d�@


accuracy_1R�?��|7       ���Y	�=v㚐�A�*)

cross_entropyr~@@


accuracy_1{.?�Ry�7       ���Y	�K�㚐�A�*)

cross_entropyV]0@


accuracy_1�Q8?c|��7       ���Y	���㚐�A�*)

cross_entropy�Q@


accuracy_1��?D=�l7       ���Y	�2�㚐�A�*)

cross_entropy!%�@


accuracy_1���>u�q7       ���Y	Y䚐�A�*)

cross_entropy���@


accuracy_1�?��(7       ���Y	{�C䚐�A�*)

cross_entropy(�@


accuracy_1���>(�C�7       ���Y	a暐�A�*)

cross_entropy�,t@


accuracy_1q=
?},�L7       ���Y	�>9暐�A�*)

cross_entropyIiz@


accuracy_1��?%��7       ���Y	E�a暐�A�*)

cross_entropy��@


accuracy_1�?Is97       ���Y	�4�暐�A�*)

cross_entropyH�i@


accuracy_1��(?\=��7       ���Y	�0�暐�A�*)

cross_entropyja.@


accuracy_1333?Uy��7       ���Y	�$�暐�A�*)

cross_entropy3)�@


accuracy_1)\?*s7       ���Y	�	皐�A�*)

cross_entropy��]@


accuracy_1�z?d�z�7       ���Y	�;3皐�A�*)

cross_entropy\g�@


accuracy_1q=
?����7       ���Y	�y]皐�A�*)

cross_entropy�*@


accuracy_1333?>���7       ���Y	ۆ皐�A�*)

cross_entropy�-@


accuracy_1
�#?���87       ���Y	G�L隐�A�*)

cross_entropy�zu@


accuracy_1�?	�@7       ���Y	dlv隐�A�*)

cross_entropyHVA


accuracy_1q=
?_꠮7       ���Y	$M�隐�A�*)

cross_entropy3Oa@


accuracy_1�?ڿ7       ���Y	.��隐�A�*)

cross_entropy�8@


accuracy_1�z?��J7       ���Y	�o�隐�A�*)

cross_entropy_+@


accuracy_1)\?�uF7       ���Y	:Ꚑ�A�*)

cross_entropy� D@


accuracy_1
�#?N8�Z7       ���Y	s�CꚐ�A�*)

cross_entropy�	^@


accuracy_1   ?�7       ���Y	+�mꚐ�A�*)

cross_entropyJ�\@


accuracy_1q=
?�Ґ7       ���Y	/��Ꚑ�A�*)

cross_entropy��W@


accuracy_1
�#?~;��7       ���Y	���Ꚑ�A�*)

cross_entropyw��?


accuracy_1�p=?v-��7       ���Y	�욐�A�*)

cross_entropy?yJ@


accuracy_1)\?���7       ���Y	�+�욐�A�*)

cross_entropy�t@


accuracy_1)\?�� I7       ���Y	�/�욐�A�*)

cross_entropy�T6@


accuracy_1R�?�7       ���Y	�:횐�A�*)

cross_entropyR$R@


accuracy_1
�#?�_�Z7       ���Y	\�+횐�A�*)

cross_entropy�G(@


accuracy_1)\?��+T7       ���Y	�U횐�A�*)

cross_entropy!�@


accuracy_1R�?0��k7       ���Y	�5횐�A�*)

cross_entropy8 @


accuracy_1{.?�#Y�7       ���Y	pe�횐�A�*)

cross_entropy��&@


accuracy_1R�?M)��7       ���Y	K(�횐�A�*)

cross_entropyM�#@


accuracy_1��(?�H�7       ���Y	}�횐�A�*)

cross_entropyT4�@


accuracy_1���>0��F7       ���Y	q���A�*)

cross_entropy�q'@


accuracy_1
�#?�f�7       ���Y	�H��A�*)

cross_entropyv@


accuracy_1   ?���s7       ���Y	{���A�*)

cross_entropy�!@


accuracy_1R�?�:��7       ���Y	�"@��A�*)

cross_entropy�c@


accuracy_1��?��i*7       ���Y	P�h��A�*)

cross_entropy�R@


accuracy_1�z?���7       ���Y	�����A�*)

cross_entropy�c@


accuracy_1)\?�?�7       ���Y	����A�*)

cross_entropy�#@


accuracy_1��(?�b@7       ���Y	p���A�*)

cross_entropy�@


accuracy_1��(?Bh�7       ���Y	����A�*)

cross_entropy�A@


accuracy_1)\?!)�7       ���Y	�;��A�*)

cross_entropy�]-@


accuracy_1���>��s7       ���Y	ç���A�*)

cross_entropyW4@


accuracy_1�z?�ʊI7       ���Y	b�'��A�*)

cross_entropyј{@


accuracy_1q=
?�)��7       ���Y	RR��A�*)

cross_entropy��?


accuracy_1��?%I�V7       ���Y	�-|��A�*)

cross_entropyOr�@


accuracy_1�?d�\7       ���Y	n���A�*)

cross_entropyt�?


accuracy_1�Q8?����7       ���Y	����A�*)

cross_entropy�s/@


accuracy_1��?T�l�7       ���Y	�����A�*)

cross_entropyJ$
@


accuracy_1��(?t���7       ���Y	�
&����A�*)

cross_entropyqi�?


accuracy_1{.?ʮC_7       ���Y	~�N����A�*)

cross_entropy�[L@


accuracy_1�?���7       ���Y	OGy����A�*)

cross_entropy N>@


accuracy_1�?����7       ���Y	� <����A�*)

cross_entropyR,�?


accuracy_1333?��G7       ���Y	�g����A�*)

cross_entropy6{�?


accuracy_1��?�˗|7       ���Y	S�����A�*)

cross_entropy��^@


accuracy_1���>���7       ���Y	ap�����A�*)

cross_entropyC>%@


accuracy_1R�?d��7       ���Y	f������A�*)

cross_entropy�%@


accuracy_1�z?�dr�7       ���Y	Y�����A�*)

cross_entropy\��?


accuracy_1��(?L֝7       ���Y	�s9����A�*)

cross_entropys�=@


accuracy_1��>^�N7       ���Y	m=c����A�*)

cross_entropyNx�?


accuracy_1�Q8?R��F7       ���Y	������A�*)

cross_entropy�(�?


accuracy_1��?�tD�7       ���Y	7t�����A�*)

cross_entropy�	$@


accuracy_1�z?��GB7       ���Y	5������A�*)

cross_entropy��@


accuracy_1��>z֢7       ���Y	#������A�*)

cross_entropyWۦ?


accuracy_1��?RP�:7       ���Y	�3�����A�*)

cross_entropyv{M@


accuracy_1�?�Q,�7       ���Y	TL ����A�*)

cross_entropy(�?


accuracy_1��?���l7       ���Y	`)����A�*)

cross_entropy�5�?


accuracy_1��(?��m�7       ���Y	oS����A�*)

cross_entropy�]
@


accuracy_1R�?j���7       ���Y	�I|����A�*)

cross_entropyNb@


accuracy_1R�?�`��7       ���Y	�b�����A�*)

cross_entropy���?


accuracy_1)\?ٶ�
7       ���Y	�������A�*)

cross_entropysc�?


accuracy_1�p=?��� 7       ���Y	4������A�*)

cross_entropy�Q@


accuracy_1��?d��57       ���Y	S������A�*)

cross_entropy�&@


accuracy_1R�?�[�_7       ���Y	&������A�*)

cross_entropy��?


accuracy_1�p=?확�7       ���Y	�����A�*)

cross_entropy6��?


accuracy_1�Q8?��^D7       ���Y	��<����A�*)

cross_entropy��?


accuracy_1
�#?�#�27       ���Y	�f����A�*)

cross_entropy��?


accuracy_1q=
?.�5�7       ���Y	|�����A�*)

cross_entropy&�@


accuracy_1)\?�t�H7       ���Y	O������A�*)

cross_entropy�G@


accuracy_1��?5	�7       ���Y	������A�*)

cross_entropye�?


accuracy_1R�?�aU�7       ���Y	�����A�*)

cross_entropye�#@


accuracy_1�z?�=�7       ���Y	p�8����A�*)

cross_entropyW��?


accuracy_1
�#?�ji�7       ���Y	pj�����A�*)

cross_entropyqOL@


accuracy_1��?W~��7       ���Y	]�' ���A�*)

cross_entropy9�@


accuracy_1��?�l-�7       ���Y	G8Q ���A�*)

cross_entropy\��?


accuracy_1
�#??1��7       ���Y	�Xz ���A�*)

cross_entropy���?


accuracy_1{.?JOOs7       ���Y	^_� ���A�*)

cross_entropy	L@


accuracy_1   ?=L��7       ���Y	k\� ���A�*)

cross_entropyсo?


accuracy_1R�?w<�7       ���Y	e�� ���A�*)

cross_entropyv��?


accuracy_1�z?+o�87       ���Y	��'���A�*)

cross_entropyZ'�?


accuracy_1
�#?�+��7       ���Y	��Q���A�*)

cross_entropyA�@


accuracy_1��?qD:�7       ���Y	 pz���A�*)

cross_entropyq�@


accuracy_1R�?ic7       ���Y	1eC���A�*)

cross_entropy���?


accuracy_1
�#?iTy�7       ���Y	�n���A�*)

cross_entropy
�?


accuracy_1��(?����7       ���Y	�����A�*)

cross_entropyĻ@


accuracy_1�z?�;�<7       ���Y	�����A�*)

cross_entropy�,�?


accuracy_1q=
?��!�7       ���Y	9�����A�*)

cross_entropy\j@


accuracy_1)\?�_�7       ���Y	�����A�*)

cross_entropy�?


accuracy_1)\?&��47       ���Y	�>���A�*)

cross_entropy���?


accuracy_1R�?�T_7       ���Y	�e���A�*)

cross_entropy� �?


accuracy_1333?��7       ���Y	%����A�*)

cross_entropyX�?


accuracy_1333?��n�7       ���Y	�����A�*)

cross_entropy��?


accuracy_1���>��7       ���Y	�����A�*)

cross_entropy]�?


accuracy_1�z?���{7       ���Y	h����A�*)

cross_entropy84�?


accuracy_1�z?4��R7       ���Y	W�����A�*)

cross_entropy���?


accuracy_1q=
?��r�7       ���Y	�k����A�*)

cross_entropy4"�?


accuracy_1   ?��7       ���Y	9j&���A�*)

cross_entropy���?


accuracy_1   ?���7       ���Y	�lP���A�*)

cross_entropyf��?


accuracy_1��?�,|7       ���Y	��z���A�*)

cross_entropy���?


accuracy_1�z?�T �7       ���Y	������A�*)

cross_entropy��?


accuracy_1�Q8?3̿7       ���Y	Cj����A�*)

cross_entropy�w�?


accuracy_1��?�8/�7       ���Y	Z����A�*)

cross_entropy��?


accuracy_1���>���7       ���Y	�d�	���A�*)

cross_entropy�ԑ?


accuracy_1��?��H7       ���Y	{��	���A�*)

cross_entropyL��?


accuracy_1
�#?>*�7       ���Y	@c
���A�*)

cross_entropy�?


accuracy_1q=
?$�$17       ���Y	�|A
���A�*)

cross_entropyicp?


accuracy_1�z?�rd7       ���Y	ڨl
���A�*)

cross_entropy ��?


accuracy_1
�#?����7       ���Y	��
���A�*)

cross_entropy��?


accuracy_1
�#?��7       ���Y	V��
���A�*)

cross_entropy�Qg?


accuracy_1\�B?R̃�7       ���Y	uz�
���A�*)

cross_entropy�?


accuracy_1�z?���#7       ���Y	�i���A�*)

cross_entropy��?


accuracy_1
�#?�>�7       ���Y	m[B���A�*)

cross_entropyas�?


accuracy_1   ?���7       ���Y	�?���A�*)

cross_entropy��@


accuracy_1�z?��F�7       ���Y	J4-���A�*)

cross_entropyaz�?


accuracy_1333?�h0>7       ���Y	�W���A�*)

cross_entropy��?


accuracy_1�z?����7       ���Y	������A�*)

cross_entropye�?


accuracy_1)\?���7       ���Y	:�����A�*)

cross_entropy���?


accuracy_1R�?�ML�7       ���Y	�����A�*)

cross_entropy6�?


accuracy_1��?U��7       ���Y	%�����A�*)

cross_entropy)��?


accuracy_1��(?��2X7       ���Y	��(���A�*)

cross_entropy�q?


accuracy_1333?e��7       ���Y	��S���A�*)

cross_entropy6h6?


accuracy_1\�B?i׉7       ���Y	{���A�*)

cross_entropy]Յ?


accuracy_1��?��Y7       ���Y	ȹP���A�*)

cross_entropy�j{?


accuracy_1
�#?1t�7       ���Y	�E{���A�*)

cross_entropy��?


accuracy_1�?/B9q7       ���Y	������A�*)

cross_entropyϗ@


accuracy_1�?Þ�7       ���Y	�����A�*)

cross_entropy��?


accuracy_1���>=�<7       ���Y	^y����A�*)

cross_entropy��?


accuracy_1��(?�i��7       ���Y	��#���A�*)

cross_entropyq?�?


accuracy_1)\?Y���7       ���Y	�N���A�*)

cross_entropy3�U?


accuracy_1
�#?I*��7       ���Y	sw���A�*)

cross_entropy4�?


accuracy_1q=
?(.�.7       ���Y	�����A�*)

cross_entropyA9�?


accuracy_1��(?h�B`7       ���Y	�i����A�*)

cross_entropy2��?


accuracy_1��(?���7       ���Y	w����A�*)

cross_entropy.(�?


accuracy_1   ?���07       ���Y	V,����A�*)

cross_entropy�@


accuracy_1��?�KP�7       ���Y	������A�*)

cross_entropy��?


accuracy_1�z?J�}�7       ���Y	m���A�*)

cross_entropy[!�?


accuracy_1��?����7       ���Y	�~@���A�*)

cross_entropy�v�?


accuracy_1q=
?<ܾ�7       ���Y	#3j���A�*)

cross_entropy���?


accuracy_1=
�>9�l7       ���Y	�S����A�*)

cross_entropy�;�?


accuracy_1���>��7       ���Y	5�����A�*)

cross_entropy��?


accuracy_1�z?��� 7       ���Y	_j����A�*)

cross_entropya8w?


accuracy_1��?",��7       ���Y	*����A�*)

cross_entropy;�?


accuracy_1�z?a��7       ���Y	W�����A�*)

cross_entropy6�?


accuracy_1�z?�&67       ���Y	���A�*)

cross_entropy��?


accuracy_1��?�'�.7       ���Y	��,���A�*)

cross_entropyZN�?


accuracy_1���>�߹�7       ���Y	+V���A�*)

cross_entropy��v?


accuracy_1��?���E7       ���Y	���A�*)

cross_entropy�w�@


accuracy_1R�?�=}L7       ���Y	������A�*)

cross_entropy}�C?


accuracy_1
�#?�з�7       ���Y	������A�*)

cross_entropy�X�?


accuracy_1)\?�UY7       ���Y	������A�*)

cross_entropy�g?


accuracy_1R�?�7       ���Y	�3$���A�*)

cross_entropya�?


accuracy_1R�?-�R-7       ���Y	��M���A�*)

cross_entropy�J�?


accuracy_1)\?Pł�7       ���Y	�����A�*)

cross_entropyr�?


accuracy_1R�?�6�+7       ���Y	j�@���A�*)

cross_entropyx?


accuracy_1R�?�:�`7       ���Y	��i���A�*)

cross_entropy�,X?


accuracy_1R�?�b��7       ���Y	�����A�*)

cross_entropy$*m?


accuracy_1R�?�κ7       ���Y	̀����A�*)

cross_entropyfI�?


accuracy_1�z?��|7       ���Y	
�����A�*)

cross_entropy�"�?


accuracy_1��(?!*�%7       ���Y	"$���A�*)

cross_entropy���?


accuracy_1��(?��:Z7       ���Y	q|<���A�*)

cross_entropyH�W?


accuracy_1
�#?�?l7       ���Y	��e���A�*)

cross_entropyі�?


accuracy_1q=
?ɦ�
7       ���Y	�����A�*)

cross_entropy{��?


accuracy_1   ?�R7       ���Y	��W���A�*)

cross_entropy�_�?


accuracy_1
�#?]�X)7       ���Y	�D����A�*)

cross_entropy���?


accuracy_1q=
?�<��7       ���Y	Ы���A�*)

cross_entropy�+�?


accuracy_1���>����7       ���Y	������A�*)

cross_entropyj�?


accuracy_1�Q8?�{��7       ���Y	Gv����A�*)

cross_entropy���?


accuracy_1q=
? Gx�7       ���Y	�{)���A�*)

cross_entropyI�n?


accuracy_1R�?��,�7       ���Y	aS���A�*)

cross_entropyT��?


accuracy_1333?"m�O7       ���Y	��}���A�*)

cross_entropy͢�?


accuracy_1�z?��''7       ���Y	P ����A�*)

cross_entropy=aZ?


accuracy_1��(?���+7       ���Y	�j����A�*)

cross_entropyq�?


accuracy_1q=
?3u�i7       ���Y	�3� ���A�*)

cross_entropyto�?


accuracy_1���>��?7       ���Y	%}� ���A�*)

cross_entropy�@�?


accuracy_1   ?�.47       ���Y	b�� ���A�*)

cross_entropy��?


accuracy_1q=
?�E% 7       ���Y	U�!���A�*)

cross_entropy: L?


accuracy_1)\? _u(7       ���Y	�6C!���A�*)

cross_entropy��L?


accuracy_1
�#?�H�7       ���Y	�lm!���A�*)

cross_entropy��?


accuracy_1�?��7       ���Y	鼖!���A�*)

cross_entropy���?


accuracy_1)\?�ú7       ���Y	}��!���A�*)

cross_entropy��?


accuracy_1333?�k9�7       ���Y	xG�!���A�*)

cross_entropytJ�?


accuracy_1��?�Q7       ���Y	Ww"���A�*)

cross_entropy�(C?


accuracy_1
�#?���7       ���Y	�9�#���A�*)

cross_entropyV�U?


accuracy_1)\?I�#�7       ���Y	�M $���A�*)

cross_entropy���?


accuracy_1�?�I�/7       ���Y	��*$���A�*)

cross_entropy��f?


accuracy_1
�#?���7       ���Y	�S$���A�*)

cross_entropy�ù?


accuracy_1���>��G7       ���Y	p�~$���A�*)

cross_entropyz	�?


accuracy_1)\?��L~7       ���Y	�b�$���A�*)

cross_entropyt3�?


accuracy_1   ?�t�C7       ���Y	�8�$���A�*)

cross_entropyC��?


accuracy_1q=
?DOW7       ���Y	O��$���A�*)

cross_entropyo��?


accuracy_1   ?񞜯7       ���Y	�H'%���A�*)

cross_entropy���?


accuracy_1�?ʴ��7       ���Y	<WT%���A�*)

cross_entropy?"�?


accuracy_1
�#?.ԯ7       ���Y	�'���A�*)

cross_entropy�|?


accuracy_1��?>���7       ���Y	N�H'���A�*)

cross_entropyX�n?


accuracy_1�z?��f�7       ���Y	��r'���A�*)

cross_entropy�?


accuracy_1q=
?s ��7       ���Y	s�'���A�*)

cross_entropy�!?


accuracy_1)\?'}�e7       ���Y	5�'���A�*)

cross_entropy�vj?


accuracy_1
�#?�p}s7       ���Y	B�'���A�*)

cross_entropy,ک?


accuracy_1)\?�&t�7       ���Y	��(���A�*)

cross_entropy&1n?


accuracy_1
�#?ܮe7       ���Y	��@(���A�*)

cross_entropy�F=?


accuracy_1�z?��?�7       ���Y	��i(���A�*)

cross_entropy���?


accuracy_1R�?�$#l7       ���Y	�7�(���A�*)

cross_entropy�t?


accuracy_1R�?��7       ���Y	}_*���A�*)

cross_entropyX�F?


accuracy_1{.?�n4�7       ���Y	h�*���A�*)

cross_entropyj�@


accuracy_1   ?�[�D7       ���Y	{E�*���A�*)

cross_entropyX�Z?


accuracy_1��?�2>7       ���Y	���*���A�*)

cross_entropy�=v?


accuracy_1
�#?�N�:7       ���Y	+���A�*)

cross_entropy�tL?


accuracy_1��?���W7       ���Y	�G-+���A�*)

cross_entropys� ?


accuracy_1�Q8?�9�7       ���Y	_5W+���A�*)

cross_entropy�!r?


accuracy_1   ?�;n-7       ���Y	M��+���A�*)

cross_entropyVl�?


accuracy_1��>��P7       ���Y	Km�+���A�*)

cross_entropy}�?


accuracy_1
�#?�:(7       ���Y	h��+���A�*)

cross_entropy�i?


accuracy_1333?�I�7       ���Y	9P�-���A�*)

cross_entropy�Q?


accuracy_1��?�nh7       ���Y	���-���A�*)

cross_entropy}��?


accuracy_1
�#?��7       ���Y	���-���A�*)

cross_entropy�1b?


accuracy_1q=
?�v}7       ���Y	O�!.���A�*)

cross_entropy<Ts?


accuracy_1�z?���x7       ���Y	&�J.���A�*)

cross_entropy��]?


accuracy_1
�#?QT�7       ���Y	�&t.���A�*)

cross_entropyg*?


accuracy_1R�?J���7       ���Y	Fb�.���A�*)

cross_entropyȡ?


accuracy_1��>)7�7       ���Y	��.���A�*)

cross_entropye�H?


accuracy_1R�?`�� 7       ���Y	ђ�.���A�*)

cross_entropyZ�w?


accuracy_1q=
?|�K7       ���Y	��/���A�*)

cross_entropyW�"@


accuracy_1�z?��@7       ���Y	���0���A�*)

cross_entropy.��?


accuracy_1�?��χ7       ���Y	�q1���A�*)

cross_entropyM��?


accuracy_1�?6$667       ���Y	i61���A�*)

cross_entropy�Uv?


accuracy_1��?��97       ���Y	�2`1���A�*)

cross_entropy��B?


accuracy_1��?���7       ���Y	���1���A�*)

cross_entropy�*�?


accuracy_1�z?n�"7       ���Y	���1���A�*)

cross_entropy�?


accuracy_1q=
?�t37       ���Y	���1���A�*)

cross_entropy��?


accuracy_1���>x���7       ���Y	l
2���A�*)

cross_entropyQuY?


accuracy_1q=
?2 7       ���Y	<W42���A�*)

cross_entropy?


accuracy_1\�B?CVJh7       ���Y	�%^2���A�*)

cross_entropy��z?


accuracy_1
�#?���h7       ���Y	��-4���A�*)

cross_entropy|G�?


accuracy_1{.?o�\7       ���Y	ÎX4���A�*)

cross_entropy��p?


accuracy_1��(?��7       ���Y	u�4���A�*)

cross_entropy� ?


accuracy_1\�B?e�N�7       ���Y	e3�4���A�*)

cross_entropy��?


accuracy_1   ?�`��7       ���Y	�{�4���A�*)

cross_entropy`o�?


accuracy_1��?�{	�7       ���Y	X&5���A�*)

cross_entropy��?


accuracy_1��?u��77       ���Y	B!+5���A�*)

cross_entropyȠ�?


accuracy_1
�#?�5;7       ���Y	V�U5���A�*)

cross_entropy�|j?


accuracy_1\�B?�f"�7       ���Y	4Z�5���A�*)

cross_entropy�[?


accuracy_1��?S)�97       ���Y	"m�5���A�*)

cross_entropy�H?


accuracy_1��?����7       ���Y	�m7���A�*)

cross_entropy!+�?


accuracy_1���>n��7       ���Y	Dr�7���A�*)

cross_entropy�?


accuracy_1��?���c7       ���Y	�U�7���A�*)

cross_entropyv��?


accuracy_1)\?iQ87       ���Y	c�7���A�*)

cross_entropyF?


accuracy_1)\?���7       ���Y	�s8���A�*)

cross_entropy�_?


accuracy_1��(?�� 7       ���Y	d?8���A�*)

cross_entropy8?


accuracy_1{.?��g@7       ���Y	G�i8���A�*)

cross_entropy;;O?


accuracy_1
�#?��@`7       ���Y	���8���A�*)

cross_entropyo<I?


accuracy_1�z?ߤ|�7       ���Y	�8���A�*)

cross_entropyb?


accuracy_1
�#?
��G7       ���Y		@�8���A�*)

cross_entropy֩?


accuracy_1��>�G>�7       ���Y	S��:���A�*)

cross_entropy`�,?


accuracy_1��(?��7       ���Y	�W�:���A�*)

cross_entropy �n?


accuracy_1��(?��@y7       ���Y	�Q;���A�*)

cross_entropy���@


accuracy_1�z?m���7       ���Y	��+;���A�*)

cross_entropy�xL?


accuracy_1
�#?rO�B7       ���Y	�TV;���A�*)

cross_entropy}�I?


accuracy_1   ?�g7       ���Y	!�;���A�*)

cross_entropy6Z8?


accuracy_1
�#?�t��7       ���Y	��;���A�*)

cross_entropy�]?


accuracy_1��(?޸�T7       ���Y	w{�;���A�*)

cross_entropy#�P?


accuracy_1��(?`!�j7       ���Y	$b�;���A�*)

cross_entropy�H?


accuracy_1{.?!L�7       ���Y	��*<���A�*)

cross_entropy��?


accuracy_1
�#?�&��7       ���Y	2��=���A�*)

cross_entropy�9?


accuracy_1�z?%�NY7       ���Y	�$>���A�*)

cross_entropy�U�?


accuracy_1�?X�1&7       ���Y	n�L>���A�*)

cross_entropy##c?


accuracy_1�z?A���7       ���Y	�v>���A�*)

cross_entropy��C?


accuracy_1
�#?@R*c7       ���Y	�+�>���A�*)

cross_entropy�pM?


accuracy_1333?W*�7       ���Y	L�>���A�*)

cross_entropy2�_?


accuracy_1R�?�vU�7       ���Y	\��>���A�*)

cross_entropy3��?


accuracy_1)\?���7       ���Y	:4?���A�*)

cross_entropy|v�?


accuracy_1�z?�7       ���Y	�G?���A�*)

cross_entropy>~�?


accuracy_1
�#?BX�!7       ���Y	��q?���A�*)

cross_entropyw�0?


accuracy_1��?�rs7       ���Y	E�7A���A�*)

cross_entropyCEI?


accuracy_1�z?�t�7       ���Y	faA���A�*)

cross_entropy;9z?


accuracy_1)\?r��7       ���Y	�A���A�*)

cross_entropy\<?


accuracy_1{.?J+�'7       ���Y	K8�A���A�*)

cross_entropy�5i?


accuracy_1R�?KpS07       ���Y	���A���A�*)

cross_entropytFW?


accuracy_1
�#?pt
7       ���Y	l�B���A�*)

cross_entropy��N?


accuracy_1
�#?	J"�7       ���Y	7�1B���A�*)

cross_entropy��f?


accuracy_1���>�q%7       ���Y	y�\B���A�*)

cross_entropy��t?


accuracy_1R�?��7       ���Y	"цB���A�*)

cross_entropyo}X?


accuracy_1
�#?s^��7       ���Y	��B���A�*)

cross_entropy=ބ?


accuracy_1333?�:�u7       ���Y	EYzD���A�*)

cross_entropy�
�?


accuracy_1q=
?�ܜ7       ���Y	Ǘ�D���A�*)

cross_entropyd�?


accuracy_1R�?qmzs7       ���Y	��D���A�*)

cross_entropy��?


accuracy_1{.?	V`7       ���Y	Qu�D���A�*)

cross_entropy�d>?


accuracy_1�z?>H�67       ���Y	/�E���A�*)

cross_entropy�#!?


accuracy_1333?�p7       ���Y	�+IE���A�*)

cross_entropy�f`?


accuracy_1�z?3=#7       ���Y	{sE���A�*)

cross_entropyE�"?


accuracy_1{.?�� 7       ���Y	�O�E���A�*)

cross_entropy��Z?


accuracy_1��?R�{7       ���Y	��E���A�*)

cross_entropy=qb?


accuracy_1
�#?'�7       ���Y	i�E���A�*)

cross_entropy�gU?


accuracy_1��?"�7       ���Y	���G���A�*)

cross_entropy��?


accuracy_1)\?:�&7       ���Y	l��G���A�*)

cross_entropy�N?


accuracy_1R�?5+=7       ���Y	T&	H���A�*)

cross_entropy�<L?


accuracy_1R�?RX^7       ���Y	t�2H���A�*)

cross_entropy,1�?


accuracy_1)\?�ӆ@7       ���Y	�\H���A�*)

cross_entropy� CB


accuracy_1�z?�By�7       ���Y	e��H���A�*)

cross_entropy� ?


accuracy_1333?��z57       ���Y	�
�H���A�*)

cross_entropy63?


accuracy_1)\?�X#�7       ���Y	��H���A�*)

cross_entropy��?


accuracy_1)\?:O�7       ���Y	EPI���A�*)

cross_entropy؁g?


accuracy_1
�#?�/u7       ���Y	UX3I���A�*)

cross_entropy|��?


accuracy_1R�?�8d7       ���Y	�3�J���A�*)

cross_entropyup�?


accuracy_1��>h���7       ���Y	�a%K���A�*)

cross_entropy=(�?


accuracy_1   ?�G��7       ���Y	l�NK���A�*)

cross_entropy��?


accuracy_1q=
?P#/�7       ���Y	NRyK���A�*)

cross_entropy�?


accuracy_1��?8���7       ���Y	�i�K���A�*)

cross_entropy��{?


accuracy_1�z?d؃�7       ���Y	���K���A�*)

cross_entropy���?


accuracy_1��?꺴U7       ���Y	]�K���A�*)

cross_entropy��?


accuracy_1R�?P�-7       ���Y	0�!L���A�*)

cross_entropy*Aw?


accuracy_1�Q8?Z��.7       ���Y	!ALL���A�*)

cross_entropyV??


accuracy_1��?�.��7       ���Y	ƮuL���A�*)

cross_entropy�K�?


accuracy_1   ?GdzB7       ���Y	[h>N���A�*)

cross_entropyD1?


accuracy_1�p=?��*7       ���Y	yEiN���A�*)

cross_entropy�6*?


accuracy_1{.?�O�7       ���Y	1��N���A�*)

cross_entropyR��?


accuracy_1R�?0^��7       ���Y	���N���A�*)

cross_entropy%��?


accuracy_1=
�>�'K�7       ���Y	��N���A�*)

cross_entropy(�+?


accuracy_1333?���A7       ���Y	�VO���A�*)

cross_entropy�&n?


accuracy_1R�?�h7       ���Y	�:O���A�*)

cross_entropy/��?


accuracy_1�?�_��7       ���Y	"]dO���A�*)

cross_entropy�#.?


accuracy_1�Q8?-�7�7       ���Y	GY�O���A�*)

cross_entropyxO�?


accuracy_1)\?���7       ���Y	D��O���A�*)

cross_entropy��o?


accuracy_1
�#?��	7       ���Y	�KyQ���A�*)

cross_entropy$�N?


accuracy_1R�?�2��7       ���Y	+�Q���A�*)

cross_entropymJ|?


accuracy_1��?@�27       ���Y	f�Q���A�*)

cross_entropyw�u?


accuracy_1R�?��&�7       ���Y	�K�Q���A�*)

cross_entropye�a?


accuracy_1{.?1��7       ���Y	� R���A�*)

cross_entropyu9?


accuracy_1R�?:F� 7       ���Y	)cJR���A�*)

cross_entropyʀ?


accuracy_1��?G���7       ���Y	�4uR���A�*)

cross_entropy�wE?


accuracy_1
�#?�7       ���Y	).�R���A�*)

cross_entropy	:?


accuracy_1
�#?�r�7       ���Y	���R���A�*)

cross_entropy�&�?


accuracy_1�z?D��-7       ���Y	)�R���A�*)

cross_entropyj0�?


accuracy_1   ?��7       ���Y	��T���A�*)

cross_entropy��o?


accuracy_1q=
?C��7       ���Y	���T���A�*)

cross_entropy�Fs?


accuracy_1R�?Wo7       ���Y	�^U���A�*)

cross_entropy�Ri?


accuracy_1
�#?D�g7       ���Y	�r>U���A�*)

cross_entropy�vt?


accuracy_1�z?$T 7       ���Y	�FgU���A�*)

cross_entropy�TB


accuracy_1�?��/�7       ���Y	;ȐU���A�*)

cross_entropyᙊ?


accuracy_1333?=RY�7       ���Y	u/�U���A�*)

cross_entropy��!?


accuracy_1{.?):7       ���Y	l��U���A�*)

cross_entropy.s?


accuracy_1
�#?��q7       ���Y	�V���A�*)

cross_entropy�/?


accuracy_1R�?o���7       ���Y	B 6V���A�*)

cross_entropy�#5?


accuracy_1�z?t��7       ���Y	��W���A�*)

cross_entropy��n?


accuracy_1333?��57       ���Y	�(X���A�*)

cross_entropyCde?


accuracy_1�z?Gg_t7       ���Y	��PX���A�*)

cross_entropy[_`?


accuracy_1���>Ԃ��7       ���Y	��yX���A�*)

cross_entropy���?


accuracy_1
�#?N��G7       ���Y	�Z�X���A�*)

cross_entropy%jb?


accuracy_1{.?���7       ���Y	���X���A�*)

cross_entropy�K_?


accuracy_1��?�҈e7       ���Y	c{�X���A�*)

cross_entropyH�?


accuracy_1R�?�6tb7       ���Y	�WY���A�*)

cross_entropy?'t?


accuracy_1���>��7       ���Y	��IY���A�*)

cross_entropy�_?


accuracy_1)\?_��@7       ���Y	�tsY���A�*)

cross_entropy3�V?


accuracy_1)\?�b��7       ���Y	5p=[���A�*)

cross_entropy��m?


accuracy_1R�?gV��7       ���Y	�0f[���A�*)

cross_entropy�p?


accuracy_1��(?q�%�7       ���Y	�Ə[���A�*)

cross_entropy739?


accuracy_1R�?'�fk7       ���Y	�t�[���A�*)

cross_entropy�~~?


accuracy_1��(?jC}>7       ���Y	��[���A�*)

cross_entropy�3�?


accuracy_1q=
?0��7       ���Y	̼\���A�*)

cross_entropy�T�?


accuracy_1q=
?LGL�7       ���Y	�5\���A�*)

cross_entropy[?


accuracy_1q=
?��D7       ���Y	��^\���A�*)

cross_entropy+��?


accuracy_1)\?�m�H7       ���Y	"�\���A�*)

cross_entropyL�*?


accuracy_1333?eZU7       ���Y	�/�\���A�*)

cross_entropy�]q?


accuracy_1)\?6�ܬ7       ���Y	Za^���A�*)

cross_entropy�^?


accuracy_1)\?�V��7       ���Y	RŨ^���A�*)

cross_entropyF�9?


accuracy_1R�?�+�_7       ���Y	��^���A�*)

cross_entropyӈ>?


accuracy_1
�#?�~��7       ���Y	��^���A�*)

cross_entropy8�@


accuracy_1�z?*ϕ�7       ���Y	�%_���A�*)

cross_entropy� +?


accuracy_1R�?Ӳ|�7       ���Y	�N_���A�*)

cross_entropy%�H?


accuracy_1)\?\�$�7       ���Y	�w_���A�*)

cross_entropy�%?


accuracy_1R�?�+�	7       ���Y	��_���A�*)

cross_entropy0�?


accuracy_1�z?,��(7       ���Y	��_���A�*)

cross_entropy��=?


accuracy_1��?OJR�7       ���Y	l��_���A�*)

cross_entropyn:6?


accuracy_1
�#?$Ő�7       ���Y	��a���A�*)

cross_entropyQ�4?


accuracy_1�z?M�v7       ���Y	_e�a���A�*)

cross_entropy�h�?


accuracy_1R�?�Ǆ�7       ���Y	�b���A�*)

cross_entropy�'e?


accuracy_1��?Y�57       ���Y	B�<b���A�*)

cross_entropy~?


accuracy_1   ?��	7       ���Y	�Ngb���A�*)

cross_entropy��J?


accuracy_1
�#?~��7       ���Y	���b���A�*)

cross_entropy$�i?


accuracy_1q=
?��\�7       ���Y	�S�b���A�*)

cross_entropy�y?


accuracy_1\�B?L��]7       ���Y	y �b���A�*)

cross_entropy&f?


accuracy_1�G�>����7       ���Y	c���A�*)

cross_entropy�*?


accuracy_1��?Aʲ 7       ���Y	X89c���A�*)

cross_entropyΛ?


accuracy_1��Q?]8��7       ���Y	]c�d���A�*)

cross_entropyCV+?


accuracy_1�p=?�db~7       ���Y	!�$e���A�*)

cross_entropyWM8?


accuracy_1��?�lE7       ���Y	�Me���A�*)

cross_entropy)V?


accuracy_1q=
?b��7       ���Y	�,xe���A�*)

cross_entropy�?


accuracy_1R�?F��7       ���Y	�@�e���A�*)

cross_entropy��N?


accuracy_1��(?��om7       ���Y	��e���A�*)

cross_entropy���?


accuracy_1)\?q�b�7       ���Y	��e���A�*)

cross_entropyf �?


accuracy_1��>58�7       ���Y	�2!f���A�*)

cross_entropy�ɝ?


accuracy_1��?�hTq7       ���Y	�jKf���A�*)

cross_entropyR�+?


accuracy_1�Q8?Ɍ~7       ���Y	r{tf���A�*)

cross_entropy	��?


accuracy_1�z?��l7       ���Y	��8h���A�*)

cross_entropy%n�?


accuracy_1q=
?���7       ���Y	�ah���A�*)

cross_entropy9��?


accuracy_1�z?X^�7       ���Y	!^�h���A�*)

cross_entropy-�)?


accuracy_1�z?��c7       ���Y	��h���A�*)

cross_entropy��C@


accuracy_1�z?�*Ң7       ���Y	w��h���A�*)

cross_entropy�@


accuracy_1R�?�j��7       ���Y	��i���A�*)

cross_entropy�Q?


accuracy_1��?���7       ���Y	��6i���A�*)

cross_entropy��?


accuracy_1q=
?�!��7       ���Y	 Aai���A�*)

cross_entropy�D@


accuracy_1�?���E7       ���Y	��i���A�*)

cross_entropy�ӹ?


accuracy_1333?�]j�7       ���Y	�>�i���A�*)

cross_entropy7�O?


accuracy_1��(?���?7       ���Y	K�k���A�*)

cross_entropyr?


accuracy_1�z?��I7       ���Y	��k���A�*)

cross_entropyJ.?


accuracy_1
�#?�=��7       ���Y	�#�k���A�*)

cross_entropyD��?


accuracy_1R�?Y�ą7       ���Y	2Tl���A�*)

cross_entropy�?


accuracy_1333?��7       ���Y	��+l���A�*)

cross_entropy=�C?


accuracy_1�?2�7       ���Y	��Vl���A�*)

cross_entropy�Xt?


accuracy_1���>%���7       ���Y	Ɓl���A�*)

cross_entropy�R?


accuracy_1��(?��'�7       ���Y	7��l���A�*)

cross_entropyM+2?


accuracy_1R�?�ׄ;7       ���Y	e�l���A�*)

cross_entropyFO~?


accuracy_1��? ��7       ���Y	���l���A�*)

cross_entropy}�U?


accuracy_1)\?��s 7       ���Y	˨�n���A�*)

cross_entropy�O?


accuracy_1�?� �S7       ���Y	r��n���A�*)

cross_entropy�Z3?


accuracy_1
�#?`�܀7       ���Y	��o���A�*)

cross_entropyc?


accuracy_1��?�Yx7       ���Y	z�Go���A�*)

cross_entropy�?


accuracy_1333?�>;v7       ���Y	�oo���A�*)

cross_entropy�c?


accuracy_1R�?wNJ�7       ���Y	�+�o���A�*)

cross_entropy�_(?


accuracy_1{.?Eئ7       ���Y	�
�o���A�*)

cross_entropy��U?


accuracy_1R�?��7       ���Y	��o���A�*)

cross_entropy26$?


accuracy_1��(?�e�7       ���Y	�p���A�*)

cross_entropy��9?


accuracy_1333?M�OL7       ���Y	�?p���A�*)

cross_entropy�<??


accuracy_1\�B?�c�q7       ���Y	�r���A�*)

cross_entropy��y?


accuracy_1)\?9�A�7       ���Y	4�4r���A�*)

cross_entropy)9O?


accuracy_1��?��i	7       ���Y	�6^r���A�*)

cross_entropy)W?


accuracy_1
�#?E'7       ���Y	��r���A�*)

cross_entropy�G?


accuracy_1{.?���7       ���Y	�Ͱr���A�*)

cross_entropyl�?


accuracy_1��?�	2l7       ���Y	���r���A�*)

cross_entropy�j?


accuracy_1��>h���7       ���Y	W�s���A�*)

cross_entropy�?


accuracy_1�?L�
7       ���Y	4*s���A�*)

cross_entropyJ,?


accuracy_1��?Ͳ�"7       ���Y	�sSs���A�*)

cross_entropy�1?


accuracy_1q=
?���7       ���Y	��~s���A�*)

cross_entropy_�=?


accuracy_1R�?S�7       ���Y	EFu���A�*)

cross_entropy2*^?


accuracy_1R�?pĚ7       ���Y	 �nu���A�*)

cross_entropy@�1?


accuracy_1R�?%SW�7       ���Y	�ݗu���A�*)

cross_entropy.3J?


accuracy_1R�?���7       ���Y	Ϛ�u���A�*)

cross_entropy�0?


accuracy_1�z?�;7       ���Y	�p�u���A�*)

cross_entropy�*=?


accuracy_1�z?��%7       ���Y	�\v���A�*)

cross_entropyrl0?


accuracy_1
�#?f�If7       ���Y	1=>v���A�*)

cross_entropy�O?


accuracy_1q=
?c�7       ���Y	~�hv���A�*)

cross_entropy���?


accuracy_1�z?4s�7       ���Y	��v���A�*)

cross_entropy]�F?


accuracy_1
�#?�}U7       ���Y	��v���A�*)

cross_entropynK@


accuracy_1�z?��y7       ���Y	��x���A�*)

cross_entropy��?


accuracy_1=
�>����7       ���Y	�x���A�*)

cross_entropy%a>?


accuracy_1{.?�$I�7       ���Y	��x���A�*)

cross_entropy��%?


accuracy_1
�#?��57       ���Y	��y���A�*)

cross_entropyCx.?


accuracy_1333?�l��7       ���Y	��-y���A�*)

cross_entropyQ�^?


accuracy_1q=
?�<T7       ���Y	�kXy���A�*)

cross_entropyA 1?


accuracy_1R�?�,��7       ���Y	�
�y���A�*)

cross_entropy�	%?


accuracy_1{.?3[�7       ���Y	���y���A�*)

cross_entropy��(?


accuracy_1��(?{��7       ���Y	t��y���A�*)

cross_entropy)n?


accuracy_1�z?@*u7       ���Y	�#z���A�*)

cross_entropya@#?


accuracy_1
�#?�7       ���Y	���{���A�*)

cross_entropy�=�?


accuracy_1��?�}7       ���Y	��{���A�*)

cross_entropy8�#?


accuracy_1��(?^��7       ���Y	��|���A�*)

cross_entropyN:?


accuracy_1R�?y�l7       ���Y	�=D|���A�*)

cross_entropy7*?


accuracy_1
�#?+�17       ���Y	��m|���A�*)

cross_entropy�$*?


accuracy_1333?��L7       ���Y	ʘ|���A�*)

cross_entropys�f?


accuracy_1��?Y[`�7       ���Y	߬�|���A�*)

cross_entropyo{.?


accuracy_1�Q8?dv�7       ���Y	k
�|���A�*)

cross_entropy�$-?


accuracy_1�z?�b�7       ���Y	:x}���A�*)

cross_entropym�S?


accuracy_1
�#?^�O�7       ���Y	�FA}���A�*)

cross_entropyC�O?


accuracy_1�z?g�/7       ���Y	b���A�*)

cross_entropy�T?


accuracy_1333?OR��7       ���Y	�6���A�*)

cross_entropy� ?


accuracy_1333?Eu�7       ���Y	�_���A�*)

cross_entropy�b�?


accuracy_1�z?���7       ���Y	Oҋ���A�*)

cross_entropy���>


accuracy_1�Q8?��c7       ���Y	L2����A�*)

cross_entropy�?


accuracy_1�G?��ʔ7       ���Y	������A�*)

cross_entropy-�	?


accuracy_1
�#?����7       ���Y	�����A�*)

cross_entropy�?


accuracy_1�G?gߴ�7       ���Y	/�5����A�*)

cross_entropy
�&?


accuracy_1��?-#�C7       ���Y	��`����A�*)

cross_entropy�1i?


accuracy_1��?�&	q7       ���Y	=I�����A�*)

cross_entropy�82?


accuracy_1�z?TJ�17       ���Y	~�a����A�*)

cross_entropyD�?


accuracy_1333?�}R�7       ���Y	_������A�*)

cross_entropy���@


accuracy_1�?�X87       ���Y	U�����A�*)

cross_entropy�R?


accuracy_1�p=?��I�7       ���Y	*߂���A�*)

cross_entropyE�7?


accuracy_1q=
?\�w7       ���Y	�}����A�*)

cross_entropyH�)?


accuracy_1��(?6�47       ���Y	Ƹ0����A�*)

cross_entropy�?


accuracy_1333?3V_�7       ���Y	`QZ����A�*)

cross_entropy.j9?


accuracy_1R�?PrK�7       ���Y	�������A�*)

cross_entropy7�i?


accuracy_1�G�>�=��7       ���Y	׭�����A�*)

cross_entropy�>H?


accuracy_1\�B?�7       ���Y	�փ���A�*)

cross_entropy6?


accuracy_1��(?�I�7       ���Y	(�����A�*)

cross_entropy8;Z?


accuracy_1R�?K���7       ���Y	b΅���A�*)

cross_entropy??


accuracy_1��?Ldo�7       ���Y	Gx�����A�*)

cross_entropy���?


accuracy_1R�?R�k7       ���Y	� ����A�*)

cross_entropyC\,?


accuracy_1
�#?��zt7       ���Y	�J����A�*)

cross_entropy��f?


accuracy_1��?*H�7       ���Y	5�s����A�*)

cross_entropy�69?


accuracy_1
�#?�7       ���Y	R_�����A�*)

cross_entropy*�$?


accuracy_1{.?�<7       ���Y	�(Ɔ���A�*)

cross_entropy��g?


accuracy_1R�?��`�7       ���Y	e������A�*)

cross_entropy)�2?


accuracy_1R�?�6c�7       ���Y	������A�*)

cross_entropy߻I?


accuracy_1�z?ȍQ7       ���Y	��戛��A�*)

cross_entropy�?V?


accuracy_1��(?�٦z7       ���Y	Z����A�*)

cross_entropy�?


accuracy_1
�#?�*87       ���Y	��9����A�*)

cross_entropyH]6?


accuracy_1
�#?��7       ���Y	�tb����A�*)

cross_entropy���?


accuracy_1)\?�cT7       ���Y	������A�*)

cross_entropy�0?


accuracy_1R�?A��i7       ���Y	V?�����A�*)

cross_entropyM%?


accuracy_1��?��M�7       ���Y	�U�����A�*)

cross_entropy�k?


accuracy_1�p=?��7       ���Y	7�
����A�*)

cross_entropy#f�?


accuracy_1   ?���7       ���Y	7�4����A�*)

cross_entropy%�B?


accuracy_1��(?��.7       ���Y	|�a����A�*)

cross_entropyrT�?


accuracy_1���>��.7       ���Y	
�(����A�*)

cross_entropy�L?


accuracy_1{.?�p.�7       ���Y	�R����A�*)

cross_entropyȠ(?


accuracy_1��?O?&G7       ���Y	$�z����A�*)

cross_entropy��?


accuracy_1�Q8?u;E�7       ���Y	�󣌛��A�*)

cross_entropy|)(?


accuracy_1)\?��N�7       ���Y	�v͌���A�*)

cross_entropy�?


accuracy_1
�#?�� �7       ���Y	�x�����A�*)

cross_entropy�5?


accuracy_1{.?�1��7       ���Y	�!����A�*)

cross_entropy8�4?


accuracy_1��(?t��Y7       ���Y	L����A�*)

cross_entropy�t'?


accuracy_1{.?B�~7       ���Y	�y����A�*)

cross_entropy|H??


accuracy_1{.?Gڣ7       ���Y	�7�����A�*)

cross_entropy$�?


accuracy_1�z?91��7       ���Y	��e����A�*)

cross_entropy�@?


accuracy_1��?�P�7       ���Y	O�����A�*)

cross_entropya�?


accuracy_1{.?����7       ���Y	(ʺ����A�*)

cross_entropy4�L?


accuracy_1q=
?� re7       ���Y	�菛��A�*)

cross_entropy
W�?


accuracy_1R�?�9��7       ���Y	�*����A�*)

cross_entropyݼ7?


accuracy_1�z?�(i7       ���Y		sB����A�*)

cross_entropy$QW?


accuracy_1
�#?���7       ���Y	Xm����A�*)

cross_entropyn9?


accuracy_1
�#?�:�7       ���Y	���A�*)

cross_entropy35A?


accuracy_1{.?�
�E7       ���Y	ҕ���A�*)

cross_entropy�>K?


accuracy_1{.?�!K�7       ���Y	�U쐛��A�*)

cross_entropy7�6?


accuracy_1��?�L"7       ���Y	V������A�*)

cross_entropy�/?


accuracy_1
�#?�g�7       ���Y	ʈݒ���A�*)

cross_entropy2T?


accuracy_1��(?��3M7       ���Y	4�����A�*)

cross_entropyIZa?


accuracy_1q=
?��_�7       ���Y	x�1����A�*)

cross_entropy��W?


accuracy_1R�?�dI�7       ���Y	�@\����A�*)

cross_entropy�?


accuracy_1\�B?��7       ���Y	�w�����A�*)

cross_entropyFNA?


accuracy_1
�#?x��f7       ���Y	|f�����A�*)

cross_entropy F?


accuracy_1��(?Cd��7       ���Y	��ٓ���A�*)

cross_entropy�a?


accuracy_1���>x�D7       ���Y	������A�*)

cross_entropys�R?


accuracy_1)\?��t57       ���Y	O�-����A�*)

cross_entropy?�Y?


accuracy_1��(?����7       ���Y	]�󕛐�A�*)

cross_entropy_c	?


accuracy_1�Q8?f���7       ���Y	������A�*)

cross_entropy��V?


accuracy_1��?��37       ���Y	��G����A�*)

cross_entropy�D?


accuracy_1   ?kª�7       ���Y	�zq����A�*)

cross_entropy�?


accuracy_1{.?��(7       ���Y	?������A�*)

cross_entropy�8?


accuracy_1�Q8?��7       ���Y	��Ė���A�*)

cross_entropy^�#?


accuracy_1
�#?Fw �7       ���Y	���A�*)

cross_entropy�?5?


accuracy_1333?irb7       ���Y	N[����A�*)

cross_entropy�9?


accuracy_1R�?�7       ���Y	�C����A�*)

cross_entropyBT?


accuracy_1��?�f�=7       ���Y	��m����A�*)

cross_entropy�b#?


accuracy_1�z?C؉7       ���Y	X�/����A�*)

cross_entropya?


accuracy_1
�#?�@�	7       ���Y	��\����A�*)

cross_entropyO/?


accuracy_1
�#?�=�s7       ���Y	+s�����A�*)

cross_entropy�4+?


accuracy_1�Q8?�BI�7       ���Y	�*�����A�*)

cross_entropy�J?


accuracy_1{.?��V7       ���Y	F�ߙ���A�*)

cross_entropyO=?


accuracy_1333?�Yk7       ���Y	�	����A�*)

cross_entropy�9?


accuracy_1
�#?��f�7       ���Y	xL2����A�*)

cross_entropy�?


accuracy_1��(?��7       ���Y	R�[����A�*)

cross_entropy��?


accuracy_1
�#?l���7       ���Y	P������A�*)

cross_entropy֤!?


accuracy_1R�?F7�37       ���Y	�Ԯ����A�*)

cross_entropy��&?


accuracy_1333?LN��7       ���Y	L�w����A�*)

cross_entropyd�%?


accuracy_1��?�Z�7       ���Y	�d�����A�*)

cross_entropy�8!?


accuracy_1R�?H� 
7       ���Y	uJʜ���A�*)

cross_entropy��3?


accuracy_1R�?i7       ���Y	+������A�*)

cross_entropy+W?


accuracy_1{.?� V�7       ���Y		)����A�*)

cross_entropyH�?


accuracy_1��(?�C�7       ���Y	x�F����A�*)

cross_entropyq�?


accuracy_1333?���17       ���Y	��o����A�*)

cross_entropy�7?


accuracy_1R�?>�27       ���Y	nV�����A�*)

cross_entropy$-?


accuracy_1�p=?�|�=7       ���Y	F Ý���A�*)

cross_entropy��)?


accuracy_1
�#?s�;�7       ���Y	��읛��A�*)

cross_entropybD?


accuracy_1
�#?94��7       ���Y	5{�����A�*)

cross_entropy۞W?


accuracy_1�?/i7       ���Y	����A�*)

cross_entropygH!?


accuracy_1{.?݄MK7       ���Y	[����A�*)

cross_entropy�_R?


accuracy_1��?ȖSc7       ���Y	f�A����A�*)

cross_entropy�=?


accuracy_1�Q8?��&7       ���Y	��j����A�*)

cross_entropy��-?


accuracy_1333?�]a�7       ���Y	5������A�*)

cross_entropyGQ?


accuracy_1��?�R-7       ���Y	�L�����A�*)

cross_entropy�Z?


accuracy_1q=
?��L�7       ���Y	l砛��A�*)

cross_entropy�C1?


accuracy_1R�?�K�V7       ���Y	![����A�*)

cross_entropy�4S?


accuracy_1)\?��c�7       ���Y	~L=����A�*)

cross_entropy�b8?


accuracy_1
�#?Pf�e7       ���Y	]�����A�*)

cross_entropyI�J?


accuracy_1��?^�07       ���Y	�0����A�*)

cross_entropy�-?


accuracy_1
�#?���7       ���Y	��[����A�*)

cross_entropyb"?


accuracy_1\�B?�`�7       ���Y	Ƌ�����A�*)

cross_entropyc#,?


accuracy_1
�#?�DOC7       ���Y	W�����A�*)

cross_entropy��J?


accuracy_1���>���7       ���Y	\�ߣ���A�*)

cross_entropy��-?


accuracy_1R�?��t�7       ���Y	������A�*)

cross_entropy�/?


accuracy_1)\?I/��7       ���Y	k6����A�*)

cross_entropy�V?


accuracy_1�Q8?�\�c7       ���Y	�_����A�*)

cross_entropy�@P?


accuracy_1
�#?��_7       ���Y	N�����A�*)

cross_entropy0�?


accuracy_1��(?G�.7       ���Y	'�G����A�*)

cross_entropys-?


accuracy_1��(?��a�7       ���Y	L�s����A�*)

cross_entropyp{?


accuracy_1
�#?���7       ���Y	�!�����A�*)

cross_entropy�d)@


accuracy_1�Q8?�ڋ7       ���Y	u�Ǧ���A�*)

cross_entropy��6?


accuracy_1
�#?�)��7       ���Y	ʪ񦛐�A�*)

cross_entropy�y?


accuracy_1�p=?�#�7       ���Y	 �����A�*)

cross_entropym�1?


accuracy_1)\?pEFB7       ���Y	$�D����A�*)

cross_entropy��,?


accuracy_1R�?|�:7       ���Y	�to����A�*)

cross_entropy�[0?


accuracy_1333?,�7       ���Y	ҙ�����A�*)

cross_entropye�2?


accuracy_1
�#?���.7       ���Y	��ç���A�*)

cross_entropy�r7?


accuracy_1��?c���7       ���Y	�P�����A�*)

cross_entropyNC-?


accuracy_1��?��7       ���Y	������A�*)

cross_entropy��?


accuracy_1R�?A�<�7       ���Y	ђߩ���A�*)

cross_entropyk!?


accuracy_1)\?4��7       ���Y	Q�
����A�*)

cross_entropyZ�P?


accuracy_1R�?5<p7       ���Y	��4����A�*)

cross_entropy\<?


accuracy_1��?�q�7       ���Y	�Z`����A�*)

cross_entropy�?


accuracy_1�p=?G �47       ���Y	殊����A�*)

cross_entropy!�?


accuracy_1333?���I7       ���Y	�����A�*)

cross_entropySD+?


accuracy_1�z?�1r�7       ���Y	��છ��A�*)

cross_entropy��?


accuracy_1333?ٰl�7       ���Y	�	����A�*)

cross_entropyD~(?


accuracy_1
�#?>-�&7       ���Y	�֬���A�*)

cross_entropy��H?


accuracy_1��(?���7       ���Y	�b ����A�*)

cross_entropy�O)?


accuracy_1q=
?)�f�7       ���Y	��*����A�*)

cross_entropy��0?


accuracy_1��?ԓ�O7       ���Y	}yU����A�*)

cross_entropy�?


accuracy_1��(?�{T7       ���Y	?������A�*)

cross_entropy� /?


accuracy_1��(?6��~7       ���Y	�������A�*)

cross_entropy�!?


accuracy_1��(?�w�7       ���Y	H�խ���A�*)

cross_entropyN�?


accuracy_1\�B?�`�'7       ���Y	�������A�*)

cross_entropylk?


accuracy_1��(?dE�@7       ���Y	@I+����A�*)

cross_entropyF.?


accuracy_1�?�O�]7       ���Y	Q�U����A�*)

cross_entropys�?


accuracy_1q=
?�}!7       ���Y	������A�*)

cross_entropy�>?


accuracy_1q=
?��7       ���Y	40L����A�*)

cross_entropy�:?


accuracy_1�z?Cp�d7       ���Y	�y����A�*)

cross_entropyH?


accuracy_1
�#?I�dq7       ���Y	-聆���A�*)

cross_entropy=�?


accuracy_1��(?��܄7       ���Y	��Ӱ���A�*)

cross_entropy��O?


accuracy_1��(?*Mx�7       ���Y	t������A�*)

cross_entropyO�;?


accuracy_1
�#?�7       ���Y	qq&����A�*)

cross_entropye�.?


accuracy_1��(?So�7       ���Y	�O����A�*)

cross_entropy&�#?


accuracy_1{.?��7       ���Y	y����A�*)

cross_entropy�'?


accuracy_1�z?F�.�7       ���Y	{Ϣ����A�*)

cross_entropy�[?


accuracy_1��?�vbM7       ���Y	�Fl����A�*)

cross_entropy�\?


accuracy_1��(?p]�7       ���Y	q������A�*)

cross_entropyܩL?


accuracy_1{.?�m^7       ���Y	��³���A�*)

cross_entropy�,$?


accuracy_1�Q8?F_{�7       ���Y	C�볛��A�*)

cross_entropy�{$?


accuracy_1
�#?I$�*7       ���Y	������A�*)

cross_entropy�S>?


accuracy_1�z?�_��7       ���Y	�g?����A�*)

cross_entropy�/G?


accuracy_1��?��X�7       ���Y	�bi����A�*)

cross_entropy?{?


accuracy_1R�?Ag3�7       ���Y	�p�����A�*)

cross_entropy�?


accuracy_1{.?_�*�7       ���Y	�}�����A�*)

cross_entropyh3?


accuracy_1{.?�f�/7       ���Y	_�紛��A�*)

cross_entropy�G?


accuracy_1��(?S�
D7       ���Y	�������A�*)

cross_entropy��&?


accuracy_1
�#?Sz�7       ���Y	*6ݶ���A�*)

cross_entropy��6?


accuracy_1q=
?x�!�7       ���Y	=^����A�*)

cross_entropy�?


accuracy_1��L?�K��7       ���Y	�n/����A�*)

cross_entropyM�.?


accuracy_1
�#?����7       ���Y	J�X����A�*)

cross_entropy��-?


accuracy_1�z?��$T7       ���Y	�@�����A�*)

cross_entropy}57?


accuracy_1R�?f���7       ���Y	r������A�*)

cross_entropyc?


accuracy_1R�?��V�7       ���Y	�WԷ���A�*)

cross_entropy��?


accuracy_1{.?�y��7       ���Y	�G�����A�*)

cross_entropy�y+?


accuracy_1R�?�6Υ7       ���Y	(����A�*)

cross_entropyi�$?


accuracy_1�z?��X7       ���Y	�𹛐�A�*)

cross_entropy�zy?


accuracy_1R�?�n�7       ���Y	������A�*)

cross_entropy�?


accuracy_1R�?JN%7       ���Y	��B����A�*)

cross_entropy�c	?


accuracy_1�Q8?$���7       ���Y	͜l����A�*)

cross_entropyc�?


accuracy_1�z?]n�-7       ���Y	/Q�����A�*)

cross_entropy��.?


accuracy_1�z?jΞP7       ���Y	�������A�*)

cross_entropy#=?


accuracy_1��(?��+�7       ���Y	��躛��A�*)

cross_entropy�S+?


accuracy_1{.?��T7       ���Y	�4����A�*)

cross_entropy6(>?


accuracy_1��(?���7       ���Y	OV<����A�*)

cross_entropy$�Q?


accuracy_1�?B*�7       ���Y	f����A�*)

cross_entropy	+/?


accuracy_1��(?ƪA�7       ���Y	W�3����A�*)

cross_entropyj�'?


accuracy_1333?�םA7       ���Y	9'\����A�*)

cross_entropy�.?


accuracy_1{.?@��7       ���Y	[�����A�*)

cross_entropy׎!?


accuracy_1{.?ʗbm7       ���Y	�-�����A�*)

cross_entropy�� ?


accuracy_1{.?s��7       ���Y	Νؽ���A�*)

cross_entropy�e:@


accuracy_1�z?�.,_7       ���Y	iO����A�*)

cross_entropy�!9?


accuracy_1q=
?#>�7       ���Y	/,����A�*)

cross_entropy�?


accuracy_1��(?8���7       ���Y	&W����A�*)

cross_entropy�?


accuracy_1��?C�7       ���Y	�����A�*)

cross_entropy�$?


accuracy_1{.?�X�n7       ���Y	�����A�*)

cross_entropy�X??


accuracy_1�z?�]�7       ���Y	��p����A�*)

cross_entropyʢ�>


accuracy_1�Q8?����7       ���Y	(������A�*)

cross_entropy7^.?


accuracy_1��(?�>N�7       ���Y	������A�*)

cross_entropy:;?


accuracy_1�p=?����7       ���Y	�g�����A�*)

cross_entropy��F?


accuracy_1��?�˸�7       ���Y	������A�*)

cross_entropyYb?


accuracy_1
�#?�˱|7       ���Y	�'C����A�*)

cross_entropy��#?


accuracy_1�z?�Df7       ���Y	m����A�*)

cross_entropy��?


accuracy_1��(?��t7       ���Y	�L�����A�*)

cross_entropy?;'?


accuracy_1��(?���7       ���Y	A������A�*)

cross_entropyuL?


accuracy_1{.?V��E7       ���Y	59�����A�*)

cross_entropy�,?


accuracy_1�z?�[J7       ���Y	=�Û��A�*)

cross_entropy��?


accuracy_1333?�� 7       ���Y	s��Û��A�*)

cross_entropy\ ?


accuracy_1R�?��7       ���Y	{�ě��A�*)

cross_entropy2�?


accuracy_1333?1M��7       ���Y	~�3ě��A�*)

cross_entropy8?


accuracy_1
�#?~ci7       ���Y	;8^ě��A�*)

cross_entropy�6?


accuracy_1
�#?)F�7       ���Y	���ě��A�*)

cross_entropyW�*?


accuracy_1\�B?W�~�7       ���Y	��ě��A�*)

cross_entropyt\%?


accuracy_1��?s��07       ���Y	´�ě��A�*)

cross_entropyv)�?


accuracy_1R�?Y�O7       ���Y	xLś��A�*)

cross_entropy �k?


accuracy_1)\?���7       ���Y	��0ś��A�*)

cross_entropy�7?


accuracy_1�z?/H$7       ���Y	G;�ƛ��A�*)

cross_entropy�0&?


accuracy_1{.?�8�7       ���Y	Q&Ǜ��A�*)

cross_entropy
�5?


accuracy_1��(?��7       ���Y	\"RǛ��A�*)

cross_entropyqC2?


accuracy_1�Q8?�K7       ���Y	�7~Ǜ��A�*)

cross_entropy��%?


accuracy_1
�#?�=7       ���Y	�&�Ǜ��A�*)

cross_entropy��?


accuracy_1R�?�mu7       ���Y	m��Ǜ��A�*)

cross_entropyQ�-?


accuracy_1�Q8?�z�7       ���Y	���Ǜ��A�*)

cross_entropy;??


accuracy_1�z?�)S7       ���Y	�&ț��A�*)

cross_entropy��T?


accuracy_1)\?;�2:7       ���Y	*4Pț��A�*)

cross_entropy��!?


accuracy_1333?WGp97       ���Y	B�xț��A�*)

cross_entropyj*?


accuracy_1��(?����7       ���Y	"�Dʛ��A�*)

cross_entropyc�M?


accuracy_1�z?��V7       ���Y	ڥqʛ��A�*)

cross_entropyV�?


accuracy_1{.?C��7       ���Y	?�ʛ��A�*)

cross_entropy�?


accuracy_1333?�	��7       ���Y	A��ʛ��A�*)

cross_entropyn�.?


accuracy_1333?���7       ���Y	q�ʛ��A�*)

cross_entropyi�?


accuracy_1�p=?���7       ���Y	�k˛��A�*)

cross_entropy��??


accuracy_1R�?�'�7       ���Y	o�A˛��A�*)

cross_entropy�62?


accuracy_1R�?���7       ���Y	�k˛��A�*)

cross_entropy��?


accuracy_1333?	�7       ���Y	1H�˛��A�*)

cross_entropy�[ ?


accuracy_1�G?���7       ���Y	f��˛��A�*)

cross_entropyI|�?


accuracy_1)\?w�>�7       ���Y	�O�͛��A�*)

cross_entropy�'?


accuracy_1�z?����7       ���Y	�I�͛��A�*)

cross_entropy37?


accuracy_1\�B?�vq7       ���Y	3��͛��A�*)

cross_entropyjE�>


accuracy_1��L?j�7       ���Y	��Λ��A�*)

cross_entropy�:?


accuracy_1���>:ʹ�7       ���Y	_n0Λ��A�*)

cross_entropy�8?


accuracy_1{.?�S
�7       ���Y	jGZΛ��A�*)

cross_entropy4[S?


accuracy_1R�?2T��7       ���Y	xs�Λ��A�*)

cross_entropy�f?


accuracy_1{.?��T�7       ���Y	q��Λ��A�*)

cross_entropy~?


accuracy_1�z?�?��7       ���Y	U��Λ��A�*)

cross_entropyƱZ?


accuracy_1
�#?��7       ���Y	՟�Λ��A�*)

cross_entropyt
?


accuracy_1
�#?�	r}7       ���Y	� �Л��A�*)

cross_entropy(g?


accuracy_1�p=?�f�*7       ���Y	8^�Л��A�*)

cross_entropy�[X?


accuracy_1{.?���I7       ���Y	��ћ��A�*)

cross_entropys%?


accuracy_1�z?`�)�7       ���Y	�|Gћ��A�*)

cross_entropy-?


accuracy_1333?ac��7       ���Y	Thqћ��A�*)

cross_entropy[�?


accuracy_1��(?r�7       ���Y	���ћ��A�*)

cross_entropyѨ1?


accuracy_1��(?�8��7       ���Y	�u�ћ��A�*)

cross_entropyQ�k?


accuracy_1�z?��ٌ7       ���Y	f��ћ��A�*)

cross_entropyH*9?


accuracy_1��(?!H�7       ���Y	2�қ��A�*)

cross_entropy�!?


accuracy_1R�?�}W{7       ���Y	��Gқ��A�*)

cross_entropy�*?


accuracy_1
�#?�n87       ���Y	�ԛ��A�*)

cross_entropy�V?


accuracy_1333?�-0<7       ���Y	\4ԛ��A�*)

cross_entropy�>?


accuracy_1��L?D�j�7       ���Y	�5]ԛ��A�*)

cross_entropy��?


accuracy_1333?��)�7       ���Y	:Ćԛ��A�*)

cross_entropy�8
?


accuracy_1��L?����7       ���Y	�~�ԛ��A�*)

cross_entropyWs ?


accuracy_1�Q8?��?B7       ���Y	���ԛ��A�*)

cross_entropy��%?


accuracy_1\�B?3>��7       ���Y	�՛��A�*)

cross_entropyG�?


accuracy_1�p=?��+7       ���Y	�F7՛��A�*)

cross_entropyj�>?


accuracy_1
�#?�j&k7       ���Y	�V`՛��A�*)

cross_entropy�+?


accuracy_1R�?[��7       ���Y	.%�՛��A�*)

cross_entropy��%?


accuracy_1�Q8?R��u7       ���Y	�#Oכ��A�*)

cross_entropy�n?


accuracy_1��Q?�f�B7       ���Y	�yכ��A�*)

cross_entropyT�5?


accuracy_1
�#?v�cU7       ���Y	I�כ��A�*)

cross_entropy�'?


accuracy_1333?|[+�7       ���Y	%O�כ��A�*)

cross_entropyӑ?


accuracy_1R�?Fu"�7       ���Y	��כ��A�*)

cross_entropyn(
?


accuracy_1�p=?��7       ���Y	�| ؛��A�*)

cross_entropyQ�?


accuracy_1333?F���7       ���Y	]+K؛��A�*)

cross_entropy?


accuracy_1�Q8?��V�7       ���Y	piu؛��A�*)

cross_entropy��
?


accuracy_1�p=?}�d�7       ���Y	Ai�؛��A�*)

cross_entropy@3=?


accuracy_1��?�o�67       ���Y	�%�؛��A�*)

cross_entropy&B?


accuracy_1R�?�G.7       ���Y	�g�ڛ��A�*)

cross_entropy�#?


accuracy_1{.?e��y7       ���Y	m�ڛ��A�*)

cross_entropy�.?


accuracy_1�z?<�;7       ���Y	��ڛ��A�*)

cross_entropy��?


accuracy_1�p=?'�%�7       ���Y	��ۛ��A�*)

cross_entropy8?


accuracy_1\�B?�t��7       ���Y	@[6ۛ��A�*)

cross_entropy���>


accuracy_1�p=?W�7       ���Y	B]`ۛ��A�*)

cross_entropy7�?


accuracy_1R�?�΂7       ���Y	���ۛ��A�*)

cross_entropy��&?


accuracy_1�z?Q�${7       ���Y	=��ۛ��A�*)

cross_entropy:~I?


accuracy_1{.?M`�/7       ���Y	{��ۛ��A�*)

cross_entropy}��>


accuracy_1\�B?]Y7       ���Y	h�ܛ��A�*)

cross_entropy�%?


accuracy_1{.?"v;�7       ���Y	��ݛ��A�*)

cross_entropy��
?


accuracy_1\�B?�m+c7       ���Y	�f�ݛ��A�*)

cross_entropyK�?


accuracy_1��(?+-��7       ���Y	\�*ޛ��A�*)

cross_entropy�?


accuracy_1
�#?X�F�7       ���Y	��Wޛ��A�*)

cross_entropy	�%?


accuracy_1�Q8?��+�7       ���Y	@X�ޛ��A�*)

cross_entropy�Y2?


accuracy_1�Q8?'+ H7       ���Y	=�ޛ��A�*)

cross_entropyX5�>


accuracy_1��L?ը>i7       ���Y	��ޛ��A�*)

cross_entropy&�?


accuracy_1��?��Q7       ���Y	���ޛ��A�*)

cross_entropy�
!?


accuracy_1��?�ˢb7       ���Y	2)ߛ��A�*)

cross_entropy8�?


accuracy_1{.?��x�7       ���Y	��Rߛ��A�*)

cross_entropyD?


accuracy_1333?o՚J7       ���Y	�� ᛐ�A�*)

cross_entropyW
?


accuracy_1{.?8
��7       ���Y	�4Jᛐ�A�*)

cross_entropy��<?


accuracy_1
�#?��_�7       ���Y	&�sᛐ�A�*)

cross_entropy��E?


accuracy_1�p=?�B�7       ���Y	Z��ᛐ�A�*)

cross_entropy@��>


accuracy_1�Q8?@�k�7       ���Y	���ᛐ�A�*)

cross_entropyn4�>


accuracy_1�p=?�yP�7       ���Y	M��ᛐ�A�*)

cross_entropyL�E?


accuracy_1�z?�eb7       ���Y	e�⛐�A�*)

cross_entropyT�?


accuracy_1333?ZΆE7       ���Y	D�B⛐�A�*)

cross_entropy��E?


accuracy_1��(?����7       ���Y	M�l⛐�A�*)

cross_entropyq?


accuracy_1�Q8?�UT�7       ���Y	�'�⛐�A�*)

cross_entropyA^?


accuracy_1�Q8?����7       ���Y	h�`䛐�A�*)

cross_entropy�?


accuracy_1��(?yR\�7       ���Y	鿊䛐�A�*)

cross_entropyd�?


accuracy_1�Q8?��07       ���Y	�5�䛐�A�*)

cross_entropy �?


accuracy_1{.?�3�I7       ���Y	���䛐�A�*)

cross_entropyU�?


accuracy_1
�#?→7       ���Y	�囐�A�*)

cross_entropyI<:?


accuracy_1R�?�(C7       ���Y	�_1囐�A�*)

cross_entropy6�?


accuracy_1�p=?��/:7       ���Y	��[囐�A�*)

cross_entropy*)'?


accuracy_1��?=��7       ���Y	�և囐�A�*)

cross_entropy}�?


accuracy_1��?��OB7       ���Y	��囐�A�*)

cross_entropy;�&?


accuracy_1��(?6�l�7       ���Y	���囐�A�*)

cross_entropy��P?


accuracy_1�z?�v�7       ���Y	�;�盐�A�*)

cross_entropy�/?


accuracy_1��?�Ay�7       ���Y	���盐�A�*)

cross_entropy
��?


accuracy_1R�?���
7       ���Y	���盐�A�*)

cross_entropybU?


accuracy_1R�?�djK7       ���Y	h�#蛐�A�*)

cross_entropy�=?


accuracy_1��?Y.l7       ���Y	P蛐�A�*)

cross_entropy��0?


accuracy_1�p=?��t7       ���Y	�y蛐�A�*)

cross_entropy��A?


accuracy_1q=
?��7       ���Y	�0�蛐�A�*)

cross_entropy�n?


accuracy_1�Q8?a�޷7       ���Y		P�蛐�A�*)

cross_entropy��V?


accuracy_1)\?�N�7       ���Y	>(�蛐�A�*)

cross_entropyX�?


accuracy_1�p=?6��7       ���Y	3!雐�A�*)

cross_entropy?�K?


accuracy_1R�?}^��7       ���Y	�o�ꛐ�A�*)

cross_entropyl�;?


accuracy_1�Q8?b���7       ���Y	F8뛐�A�*)

cross_entropy��/?


accuracy_1
�#?��E7       ���Y	��:뛐�A�*)

cross_entropy=?


accuracy_1R�?�=�7       ���Y	:�c뛐�A�*)

cross_entropyþI?


accuracy_1)\?�:�7       ���Y	��뛐�A�*)

cross_entropy�?


accuracy_1�z?�_	7       ���Y	�d�뛐�A�*)

cross_entropy*/>?


accuracy_1333?I+'
7       ���Y	 �뛐�A�*)

cross_entropy��)?


accuracy_1
�#?e�y7       ���Y	k�원�A�*)

cross_entropy�F-?


accuracy_1
�#?�9�7       ���Y	665원�A�*)

cross_entropy�R0?


accuracy_1333?��1$7       ���Y	�`원�A�*)

cross_entropy��J?


accuracy_1{.?�X�7       ���Y	m'�A�*)

cross_entropy�$#?


accuracy_1R�?�%|7       ���Y	ЃQ�A�*)

cross_entropy�!?


accuracy_1
�#?~�jg7       ���Y	dl|�A�*)

cross_entropy�X,?


accuracy_1��(?YG�J7       ���Y	�`��A�*)

cross_entropyd�?


accuracy_1�p=?�D7       ���Y	���A�*)

cross_entropy�B?


accuracy_1�z?6��'7       ���Y	<"��A�*)

cross_entropy '9?


accuracy_1��(?Ш"%7       ���Y	��'�A�*)

cross_entropys�?


accuracy_1{.?��7       ���Y	�MR�A�*)

cross_entropyr�>?


accuracy_1��?cwj�7       ���Y	�\|�A�*)

cross_entropy�+?


accuracy_1{.?��).7       ���Y	W¦�A�*)

cross_entropy�?


accuracy_1{.?�W�?7       ���Y	upr��A�*)

cross_entropy'8?


accuracy_1
�#?f�|�7       ���Y	 ����A�*)

cross_entropyq�"?


accuracy_1333?B; 7       ���Y	^ ���A�*)

cross_entropy�y?


accuracy_1
�#?!���7       ���Y	 ����A�*)

cross_entropy��?


accuracy_1\�B?���7       ���Y	
��A�*)

cross_entropyAV?


accuracy_1��Q?"KMX7       ���Y	��G��A�*)

cross_entropy�X)?


accuracy_1�z?�v��7       ���Y	�r��A�*)

cross_entropyiK?


accuracy_1{.?m4�7       ���Y	�����A�*)

cross_entropytV?


accuracy_1{.?Lt��7       ���Y	<g���A�*)

cross_entropy}�?


accuracy_1�Q8?���7       ���Y	�����A�*)

cross_entropy�?


accuracy_1333?�\�7       ���Y	b�����A�*)

cross_entropy (?


accuracy_1333?L�27       ���Y	�#�����A�*)

cross_entropy���>


accuracy_1{.?�3fo7       ���Y	E;����A�*)

cross_entropy+� ?


accuracy_1R�?j�RP7       ���Y	�6����A�*)

cross_entropym�:?


accuracy_1��(?�M��7       ���Y	��b����A�*)

cross_entropy�7?


accuracy_1��?[q|$7       ���Y	�s�����A�*)

cross_entropy�?


accuracy_1333?� �W7       ���Y	[�����A�*)

cross_entropy�?


accuracy_1�Q8?p�E7       ���Y	e������A�*)

cross_entropy�!/?


accuracy_1�z?�r7       ���Y	������A�*)

cross_entropyI
+?


accuracy_1�z?M/�*7       ���Y	R7����A�*)

cross_entropy(?


accuracy_1333?�gv7       ���Y	)������A�*)

cross_entropy�?


accuracy_1{.?��d7       ���Y	q�(����A�*)

cross_entropy?


accuracy_1�Q8?z�47       ���Y	
�T����A�*)

cross_entropyKu?


accuracy_1��(?M��7       ���Y	~����A�*)

cross_entropy���>


accuracy_1�Q8?�i�B7       ���Y	e������A�*)

cross_entropy�4?


accuracy_1\�B?>ci7       ���Y	�������A�*)

cross_entropy��g?


accuracy_1{.?1�ʖ7       ���Y	�#�����A�*)

cross_entropy�(?


accuracy_1��(?��7       ���Y	><$����A�*)

cross_entropy��?


accuracy_1
�#?D]_7       ���Y	k�L����A�*)

cross_entropy�.?


accuracy_1�Q8?���7       ���Y	��v����A�*)

cross_entropy1�0?


accuracy_1��(?��K7       ���Y	��I����A�*)

cross_entropy�e?


accuracy_1�p=?�7�7       ���Y	�,s����A�*)

cross_entropyc�?


accuracy_1��(?A���7       ���Y	4y�����A�*)

cross_entropy�'?


accuracy_1��(?��b7       ���Y	;������A�*)

cross_entropy�"?


accuracy_1{.?@��m7       ���Y	Q�����A�*)

cross_entropy�)?


accuracy_1��?Ki;�7       ���Y	�~����A�*)

cross_entropyJ8<?


accuracy_1333??��?7       ���Y	� D����A�*)

cross_entropy7v?


accuracy_1R�?�m��7       ���Y	S�l����A�*)

cross_entropy�?


accuracy_1333?�/5�7       ���Y	6�����A�*)

cross_entropy�U+?


accuracy_1333?����7       ���Y	=������A�*)

cross_entropy��?


accuracy_1R�?>c�7       ���Y	������A�*)

cross_entropy�**?


accuracy_1{.?��'�7       ���Y	,�����A�*)

cross_entropy!V?


accuracy_1�p=?va7       ���Y	�����A�*)

cross_entropy4D?


accuracy_1{.?7       ���Y	������A�*)

cross_entropy-�,?


accuracy_1��?�&[%7       ���Y	$
5����A�*)

cross_entropycG?


accuracy_1��(?�Vr7       ���Y	g�^����A�*)

cross_entropyT�?


accuracy_1�p=?ĥ��7       ���Y	F�����A�*)

cross_entropyZ�?


accuracy_1{.?4I�?7       ���Y	�������A�*)

cross_entropyi�*?


accuracy_1
�#?�#�7       ���Y	�������A�*)

cross_entropy�M"?


accuracy_1R�?د�J7       ���Y	�� ���A�*)

cross_entropy\`?


accuracy_1\�B?`YJ�7       ���Y	S/����A�*)

cross_entropy!�>


accuracy_1�(\?��t�7       ���Y	<K����A�*)

cross_entropy�c�>


accuracy_1�G?���7       ���Y	d!���A�*)

cross_entropy9!?


accuracy_1�Q8?��567       ���Y	�J���A�*)

cross_entropyi�1?


accuracy_1
�#?�4j�7       ���Y	�3r���A�*)

cross_entropy�V0?


accuracy_1�z?�M�u7       ���Y		�����A�*)

cross_entropy���>


accuracy_1�Q8?�]7       ���Y	k2����A�*)

cross_entropy��?


accuracy_1333?q0J7       ���Y	�p����A�*)

cross_entropy�.?


accuracy_1R�?X�;�7       ���Y	����A�*)

cross_entropy�P?


accuracy_1�z? �g7       ���Y	<�I���A�*)

cross_entropy�?


accuracy_1��?\�87       ���Y	 k���A�*)

cross_entropy1�E?


accuracy_1
�#?@�@�7       ���Y	+�@���A�*)

cross_entropy��
?


accuracy_1{.?-�BF7       ���Y	>�i���A�*)

cross_entropy�?


accuracy_1��(?�)4�7       ���Y	9�����A�*)

cross_entropyF�)?


accuracy_1R�?�F�	7       ���Y	a=����A�*)

cross_entropyo�"?


accuracy_1
�#?�iT7       ���Y	xp����A�*)

cross_entropy4�?


accuracy_1�p=?��E�7       ���Y	S,���A�*)

cross_entropy��?


accuracy_1��(?�}A!7       ���Y	�*;���A�*)

cross_entropy8�?


accuracy_1��(?�f$�7       ���Y	��d���A�*)

cross_entropy�D?


accuracy_1R�?d�s�7       ���Y	�8����A�*)

cross_entropy�a?


accuracy_1�G?���z7       ���Y	�TX���A�*)

cross_entropy�"?


accuracy_1{.?����7       ���Y	�����A�*)

cross_entropyd�?


accuracy_1�p=?� �W7       ���Y	�V����A�*)

cross_entropy��*?


accuracy_1
�#?�V1�7       ���Y	������A�*)

cross_entropyV�?


accuracy_1\�B?�\I�7       ���Y	x� 	���A�*)

cross_entropyh�?


accuracy_1�p=?S(=�7       ���Y	f�*	���A�*)

cross_entropy�?


accuracy_1�p=?8��\7       ���Y	�EU	���A�*)

cross_entropy=w?


accuracy_1{.?6\�7       ���Y	��	���A�*)

cross_entropy�~?


accuracy_1{.?.~�N7       ���Y	��	���A�*)

cross_entropyĤ?


accuracy_1333?3���7       ���Y	���	���A�*)

cross_entropy�?


accuracy_1�p=?���<7       ���Y	�U����A�*)

cross_entropy��?


accuracy_1��?���7       ���Y	�W����A�*)

cross_entropy�?


accuracy_1
�#?};C%7       ���Y	�Q����A�*)

cross_entropy�� ?


accuracy_1{.?H��7       ���Y	4����A�*)

cross_entropy�&?


accuracy_1��?����7       ���Y	��B���A�*)

cross_entropy;?


accuracy_1�p=?.��7       ���Y	�>m���A�*)

cross_entropy�?


accuracy_1{.?�BW	7       ���Y	Jw����A�*)

cross_entropyO}J?


accuracy_1��(?#�Ҹ7       ���Y	�E����A�*)

cross_entropy��5?


accuracy_1�z?
�H�7       ���Y	�����A�*)

cross_entropyJ#-?


accuracy_1��?��<7       ���Y	����A�*)

cross_entropy�?


accuracy_1333?L�`7       ���Y	������A�*)

cross_entropy{�	?


accuracy_1�Q8?�,77       ���Y	s����A�*)

cross_entropy:?


accuracy_1�Q8?R5�7       ���Y	�1���A�*)

cross_entropycH"?


accuracy_1\�B?�d��7       ���Y	�Z���A�*)

cross_entropy�n ?


accuracy_1�z?io�7       ���Y	A����A�*)

cross_entropy���>


accuracy_1�Q8?.V�7       ���Y	�����A�*)

cross_entropy��5?


accuracy_1�z?H�u�7       ���Y	�t����A�*)

cross_entropy�[@?


accuracy_1��?��7       ���Y	xZ���A�*)

cross_entropy�m?


accuracy_1R�?f�)67       ���Y	�p.���A�*)

cross_entropy��?


accuracy_1\�B?�G�(7       ���Y	�([���A�*)

cross_entropy��X?


accuracy_1���>suH�7       ���Y	����A�*)

cross_entropy���>


accuracy_1�(\?4�3�7       ���Y	DH���A�*)

cross_entropy��"?


accuracy_1��?���7       ���Y	��r���A�*)

cross_entropy��?


accuracy_1333?��7       ���Y	8C����A�*)

cross_entropy�]?


accuracy_1��(?��7       ���Y	-����A�*)

cross_entropy@t?


accuracy_1{.?���7       ���Y	
����A�*)

cross_entropy�.?


accuracy_1333?ɻD7       ���Y	*#���A�*)

cross_entropyn$?


accuracy_1{.?2���7       ���Y	�M���A�*)

cross_entropy�"+?


accuracy_1R�?vGzS7       ���Y	�kv���A�*)

cross_entropy�$?


accuracy_1{.?��S7       ���Y	�����A�*)

cross_entropy0�?


accuracy_1333?	;��7       ���Y	�,g���A�*)

cross_entropyQ�?


accuracy_1�G?K 2�7       ���Y	�/����A�*)

cross_entropyN�?


accuracy_1{.?�|��7       ���Y	�����A�*)

cross_entropy�8D?


accuracy_1��?#��P7       ���Y	b�����A�*)

cross_entropy�a?


accuracy_1333?L�-c7       ���Y	�8���A�*)

cross_entropy8C?


accuracy_1�?���7       ���Y	L�;���A�*)

cross_entropy�J?


accuracy_1��(?��IM7       ���Y	[�e���A�*)

cross_entropy��7?


accuracy_1)\?'�q�7       ���Y	������A�*)

cross_entropy��	?


accuracy_1��(?g}ˬ7       ���Y	������A�*)

cross_entropyϣ?


accuracy_1333?��X7       ���Y	O����A�*)

cross_entropy}.?


accuracy_1�Q8?��L7       ���Y	�8����A�*)

cross_entropyT�?


accuracy_1333?Cˆ*7       ���Y	�.����A�*)

cross_entropy-�,?


accuracy_1{.?�$]�7       ���Y	t�����A�*)

cross_entropy�! ?


accuracy_1
�#? �>7       ���Y	�'���A�*)

cross_entropy�*?


accuracy_1R�?~D�e7       ���Y	}
Q���A�*)

cross_entropyF�?


accuracy_1�p=?(�&�7       ���Y	L{���A�*)

cross_entropyq�!?


accuracy_1{.?���7       ���Y	LJ����A�*)

cross_entropy.��>


accuracy_1�G?����7       ���Y	\<����A�*)

cross_entropy�0(?


accuracy_1R�?:\7       ���Y	�e����A�*)

cross_entropy�b2?


accuracy_1333?G�-r7       ���Y	��"���A�*)

cross_entropyl?


accuracy_1�Q8?���7       ���Y	�j����A�*)

cross_entropy�m?


accuracy_1{.?+���7       ���Y	�����A�*)

cross_entropyɨ?


accuracy_1333?�w؝7       ���Y	��D���A�	*)

cross_entropy1�?


accuracy_1�Q8?��I�7       ���Y	]�n���A�	*)

cross_entropy�?


accuracy_1�p=?���7       ���Y	�����A�	*)

cross_entropy�?


accuracy_1��(?5���7       ���Y	������A�	*)

cross_entropy��	?


accuracy_1{.?���C7       ���Y	�����A�	*)

cross_entropy��?


accuracy_1
�#?�`�m7       ���Y	|���A�	*)

cross_entropyq�?


accuracy_1�p=?�|7       ���Y	��B���A�	*)

cross_entropy��%?


accuracy_1
�#?H���7       ���Y	6�k���A�	*)

cross_entropy	Q?


accuracy_1{.?��Fr7       ���Y	F�2���A�	*)

cross_entropy��5?


accuracy_1R�?O�,�7       ���Y	+�]���A�	*)

cross_entropyb?


accuracy_1��(?�;f7       ���Y	.�����A�	*)

cross_entropy
?


accuracy_1
�#?��7       ���Y	�m����A�	*)

cross_entropy�
?


accuracy_1\�B?���7       ���Y	�$����A�	*)

cross_entropy5X?


accuracy_1�Q8?���7       ���Y	�@ ���A�	*)

cross_entropy�?


accuracy_1�G?���7       ���Y	}3 ���A�	*)

cross_entropy �9?


accuracy_1��L?��7       ���Y	��] ���A�	*)

cross_entropyU%?


accuracy_1
�#?��7       ���Y	��� ���A�	*)

cross_entropyT#?


accuracy_1{.?��9�7       ���Y	Y�� ���A�	*)

cross_entropyI�H?


accuracy_1�z?aq�W7       ���Y	I�w"���A�	*)

cross_entropy�?


accuracy_1R�?t0��7       ���Y	z��"���A�	*)

cross_entropyĂ?


accuracy_1333?�y�Z7       ���Y	��"���A�	*)

cross_entropy�l�>


accuracy_1\�B?^��7       ���Y	>�"���A�	*)

cross_entropy�x ?


accuracy_1�G?��Κ7       ���Y	.q"#���A�	*)

cross_entropy���>


accuracy_1��L?L"�7       ���Y	τM#���A�	*)

cross_entropy߹ ?


accuracy_1R�?�#��7       ���Y	�Lw#���A�	*)

cross_entropy�x#?


accuracy_1��(?�e�i7       ���Y	I�#���A�	*)

cross_entropy��?


accuracy_1333?�F��7       ���Y	�W�#���A�	*)

cross_entropyVH?


accuracy_1�Q8?mDs7       ���Y	V�#���A�	*)

cross_entropyL
?


accuracy_1\�B?��7       ���Y	���%���A�	*)

cross_entropy�P#?


accuracy_1��?`��;7       ���Y	���%���A�	*)

cross_entropy]?


accuracy_1{.?���Z7       ���Y	�}&���A�	*)

cross_entropy��!?


accuracy_1��?D-7       ���Y	��E&���A�	*)

cross_entropy�1?


accuracy_1R�?�{�7       ���Y	��n&���A�	*)

cross_entropy�I?


accuracy_1�z?Xw��7       ���Y	ɼ�&���A�	*)

cross_entropy�d?


accuracy_1��?(G�7       ���Y	;�&���A�	*)

cross_entropyB�?


accuracy_1�Q8?߲�7       ���Y	��&���A�	*)

cross_entropy��?


accuracy_1�p=?�ZT7       ���Y	݇'���A�	*)

cross_entropy7�?


accuracy_1333? ��7       ���Y	��H'���A�	*)

cross_entropy�/?


accuracy_1��(?+�7       ���Y	�)���A�	*)

cross_entropyy?


accuracy_1{.?���7       ���Y	�=)���A�	*)

cross_entropy[�?


accuracy_1333?S��7       ���Y	�gg)���A�	*)

cross_entropy�&?


accuracy_1333?�SxQ7       ���Y	�*�)���A�	*)

cross_entropy,F
?


accuracy_1333?�q4~7       ���Y	�۹)���A�	*)

cross_entropyXb�>


accuracy_1333?�@i7       ���Y	��)���A�	*)

cross_entropy<k?


accuracy_1
�#?����7       ���Y	�*���A�	*)

cross_entropy��P?


accuracy_1)\?-��$7       ���Y	&�8*���A�	*)

cross_entropyO. ?


accuracy_1�Q8?�d�7       ���Y	�c*���A�	*)

cross_entropy�8?


accuracy_1�Q8?�M7       ���Y	9�*���A�	*)

cross_entropyZ*#?


accuracy_1��(?�H_7       ���Y	�9S,���A�	*)

cross_entropyN�>


accuracy_1�p=?��7       ���Y	�U|,���A�	*)

cross_entropy�?


accuracy_1
�#?3���7       ���Y	��,���A�	*)

cross_entropy���>


accuracy_1�Q8?m�,P7       ���Y	���,���A�	*)

cross_entropyV�?


accuracy_1R�?~@67       ���Y	0��,���A�	*)

cross_entropyt]?


accuracy_1�Q8?��A7       ���Y	��$-���A�	*)

cross_entropy�?


accuracy_1{.?;Tb97       ���Y	tO-���A�	*)

cross_entropy]�?


accuracy_1�p=?S��7       ���Y	4"z-���A�	*)

cross_entropy=?


accuracy_1��(?
엤7       ���Y	4�-���A�	*)

cross_entropy�3?


accuracy_1�z?�W��7       ���Y	.�-���A�	*)

cross_entropy�� ?


accuracy_1�Q8?4��7       ���Y	�%�/���A�	*)

cross_entropy#�4?


accuracy_1��(?�X'�7       ���Y	�̾/���A�	*)

cross_entropy�Z�>


accuracy_1��L?���7       ���Y	+'�/���A�	*)

cross_entropy4u?


accuracy_1�p=?�c]S7       ���Y	t�0���A�	*)

cross_entropyӏ?


accuracy_1�Q8?�d�7       ���Y	�<0���A�	*)

cross_entropyH*?


accuracy_1��?�j�7       ���Y	}h0���A�	*)

cross_entropy��?


accuracy_1{.?��7       ���Y	�G�0���A�	*)

cross_entropy��?


accuracy_1{.?�1��7       ���Y	q2�0���A�	*)

cross_entropyP�>


accuracy_1��L?o�V)7       ���Y	tD�0���A�	*)

cross_entropyMt'?


accuracy_1
�#?Z��7       ���Y	��1���A�	*)

cross_entropy�(&?


accuracy_1�p=?G�7       ���Y	���2���A�	*)

cross_entropy��?


accuracy_1�p=? �ˀ7       ���Y	+X3���A�	*)

cross_entropyԥ ?


accuracy_1333?��os7       ���Y	s53���A�	*)

cross_entropyh�J?


accuracy_1   ?�w�7       ���Y	ѕ`3���A�	*)

cross_entropy� ?


accuracy_1R�?�X��7       ���Y	�<�3���A�	*)

cross_entropy�C%?


accuracy_1��(?���7       ���Y	�=�3���A�	*)

cross_entropy6)�>


accuracy_1�Q8?�e7       ���Y	�f�3���A�	*)

cross_entropy��?


accuracy_1{.?�|�m7       ���Y	`d
4���A�	*)

cross_entropy�X?


accuracy_1333?Sԩ�7       ���Y	�k44���A�	*)

cross_entropyH�*?


accuracy_1R�?q��7       ���Y	�s_4���A�	*)

cross_entropy?


accuracy_1
�#??���7       ���Y	�� 6���A�	*)

cross_entropyq@


accuracy_1��?]�7       ���Y	MVK6���A�	*)

cross_entropy[�%?


accuracy_1333?�v7       ���Y	��u6���A�	*)

cross_entropy=?


accuracy_1\�B?�Ȗ;7       ���Y	Hv�6���A�	*)

cross_entropy)k�>


accuracy_1��L?��
�7       ���Y	���6���A�	*)

cross_entropyM}?


accuracy_1{.?T�7       ���Y	�;�6���A�	*)

cross_entropy�
0?


accuracy_1333?W��|7       ���Y	L�7���A�	*)

cross_entropyW�?


accuracy_1��(?:@�O7       ���Y	��I7���A�	*)

cross_entropy��?


accuracy_1{.?�k��7       ���Y	��t7���A�	*)

cross_entropy��	?


accuracy_1333?��t7       ���Y	>�7���A�	*)

cross_entropy	�+?


accuracy_1{.?dK��7       ���Y	�zf9���A�	*)

cross_entropy@F$?


accuracy_1�Q8?�87       ���Y	�Ӑ9���A�	*)

cross_entropy�n?


accuracy_1R�?�ᰚ7       ���Y	�Ҽ9���A�	*)

cross_entropyI`�>


accuracy_1�G?!ƪ7       ���Y	a��9���A�	*)

cross_entropy�	?


accuracy_1�p=?��nl7       ���Y	�:���A�	*)

cross_entropy��?


accuracy_1\�B?u(?�7       ���Y	�=:���A�	*)

cross_entropy��?


accuracy_1��(?tG�17       ���Y	X�e:���A�	*)

cross_entropy:�>


accuracy_1��L?'=~	7       ���Y	�?�:���A�	*)

cross_entropy�2 ?


accuracy_1�G?r��V7       ���Y	]�:���A�	*)

cross_entropy�e?


accuracy_1�p=?=׸W7       ���Y	=��:���A�	*)

cross_entropy�/?


accuracy_1{.?�s4�7       ���Y	���<���A�	*)

cross_entropy�?


accuracy_1�p=?xe7       ���Y	�z�<���A�	*)

cross_entropy��?


accuracy_1R�?�y)7       ���Y	��=���A�	*)

cross_entropyh�)?


accuracy_1{.?�S@�7       ���Y	�{;=���A�	*)

cross_entropy��??


accuracy_1R�?�t��7       ���Y	ʜe=���A�	*)

cross_entropy:?


accuracy_1333?j	��7       ���Y	�͐=���A�	*)

cross_entropy��?


accuracy_1��(?PQ��7       ���Y	��=���A�	*)

cross_entropy��?


accuracy_1��(?�p��7       ���Y	c��=���A�	*)

cross_entropy��R?


accuracy_1�z?S��A7       ���Y	�1>���A�	*)

cross_entropy�?


accuracy_1{.?�x��7       ���Y	a9:>���A�	*)

cross_entropy(�"?


accuracy_1R�?#Ϗ7       ���Y	w[@���A�	*)

cross_entropyV��>


accuracy_1��L?'e{�7       ���Y	o/@���A�	*)

cross_entropy�X?


accuracy_1�p=?��I27       ���Y	a�X@���A�	*)

cross_entropyV�?


accuracy_1{.?6�}�7       ���Y	)�@���A�	*)

cross_entropy�*?


accuracy_1�Q8?����7       ���Y	��@���A�	*)

cross_entropy�TF?


accuracy_1333?��l7       ���Y	�&�@���A�	*)

cross_entropy�#?


accuracy_1
�#?8��7       ���Y	���@���A�	*)

cross_entropy��?


accuracy_1R�?����7       ���Y	��*A���A�	*)

cross_entropy+P!?


accuracy_1��?��X7       ���Y	�VUA���A�	*)

cross_entropye�?


accuracy_1333?�P�7       ���Y	*/�A���A�	*)

cross_entropy��4?


accuracy_1��(?��U7       ���Y	��FC���A�	*)

cross_entropy�J9?


accuracy_1��(?��|Y7       ���Y	��pC���A�	*)

cross_entropy�=!?


accuracy_1333?}�TU7       ���Y	a��C���A�	*)

cross_entropyK�?


accuracy_1��(?t��7       ���Y	�
�C���A�	*)

cross_entropy�^"?


accuracy_1R�?�67       ���Y	+1�C���A�	*)

cross_entropy_?


accuracy_1�p=?e1�y7       ���Y	f�D���A�	*)

cross_entropy��?


accuracy_1\�B?kT�7       ���Y	ImCD���A�	*)

cross_entropy��?


accuracy_1��(?b�%�7       ���Y	�<nD���A�	*)

cross_entropy�?


accuracy_1�p=?�C7       ���Y	>�D���A�	*)

cross_entropy�.�>


accuracy_1�(\?Kv�7       ���Y	;:�D���A�	*)

cross_entropy�^?


accuracy_1�Q8?_��/7       ���Y	)(�F���A�
*)

cross_entropyC'?


accuracy_1�p=?=�K7       ���Y	�H�F���A�
*)

cross_entropy�?


accuracy_1�Q8?�,c%7       ���Y	d��F���A�
*)

cross_entropy��?


accuracy_1��L?�y7       ���Y	�'
G���A�
*)

cross_entropy�E*?


accuracy_1R�?�r�A7       ���Y	JF7G���A�
*)

cross_entropy�?


accuracy_1�Q8?�<�e7       ���Y	8xbG���A�
*)

cross_entropyS� ?


accuracy_1�Q8?����7       ���Y	P�G���A�
*)

cross_entropy��?


accuracy_1�p=?��V�7       ���Y	e�G���A�
*)

cross_entropyoj?


accuracy_1333?Ux�_7       ���Y	���G���A�
*)

cross_entropy�?


accuracy_1{.?]8��7       ���Y	DOH���A�
*)

cross_entropym�?


accuracy_1�Q8?8F(d7       ���Y	���I���A�
*)

cross_entropy�?


accuracy_1�Q8?r��7       ���Y	)IJ���A�
*)

cross_entropy;�?


accuracy_1�Q8?��hW7       ���Y	m�+J���A�
*)

cross_entropy<iK?


accuracy_1��?��7�7       ���Y	�EVJ���A�
*)

cross_entropy�z?


accuracy_1\�B?�I�7       ���Y	f�J���A�
*)

cross_entropy8{?


accuracy_1�Q8?�S7       ���Y	%c�J���A�
*)

cross_entropyp?


accuracy_1
�#?�\s�7       ���Y	d��J���A�
*)

cross_entropy@(?


accuracy_1
�#?�kg�7       ���Y	�.�J���A�
*)

cross_entropy�?


accuracy_1
�#?9>Jf7       ���Y	��)K���A�
*)

cross_entropyT�?


accuracy_1�p=?B�k
7       ���Y	9vTK���A�
*)

cross_entropy��%?


accuracy_1R�?��9�7       ���Y	��M���A�
*)

cross_entropy�&�>


accuracy_1��L?��Qz7       ���Y	�}EM���A�
*)

cross_entropye
?


accuracy_1{.?��#7       ���Y	�apM���A�
*)

cross_entropy�;?


accuracy_1
�#?�2�7       ���Y	ֱ�M���A�
*)

cross_entropy��?


accuracy_1��(?�w��7       ���Y	�J�M���A�
*)

cross_entropy&�?


accuracy_1{.?j�67       ���Y	v��M���A�
*)

cross_entropy��?


accuracy_1333?=��7       ���Y	DN���A�
*)

cross_entropy��?


accuracy_1{.?,]�17       ���Y	QBN���A�
*)

cross_entropy�K?


accuracy_1)\? srI7       ���Y	�nN���A�
*)

cross_entropy��?


accuracy_1333?���j7       ���Y	�ژN���A�
*)

cross_entropyi��>


accuracy_1�G?��8H7       ���Y	��hP���A�
*)

cross_entropy��>


accuracy_1�p=?���7       ���Y	�%�P���A�
*)

cross_entropy:w�>


accuracy_1��L?��7       ���Y	[��P���A�
*)

cross_entropy�H!?


accuracy_1��?&��X7       ���Y	HT�P���A�
*)

cross_entropy@��>


accuracy_1\�B?i�h�7       ���Y	p'Q���A�
*)

cross_entropy�9?


accuracy_1\�B?j	�*7       ���Y	��<Q���A�
*)

cross_entropy.�?


accuracy_1�p=?�\�7       ���Y	:�gQ���A�
*)

cross_entropyN�?


accuracy_1\�B?���7       ���Y	<L�Q���A�
*)

cross_entropy:�>


accuracy_1��Q?�+h"7       ���Y	�]�Q���A�
*)

cross_entropyWV?


accuracy_1�Q8?u�7       ���Y	��Q���A�
*)

cross_entropyV�&?


accuracy_1R�?)7       ���Y	DƴS���A�
*)

cross_entropy6�&?


accuracy_1
�#?��7       ���Y	PN�S���A�
*)

cross_entropy�]?


accuracy_1333?�i8�7       ���Y	%+T���A�
*)

cross_entropy�5?


accuracy_1�Q8? �� 7       ���Y	��2T���A�
*)

cross_entropyc]?


accuracy_1\�B?6�6�7       ���Y	��\T���A�
*)

cross_entropy3?


accuracy_1��(?ր�7       ���Y	���T���A�
*)

cross_entropy-?


accuracy_1{.?Y}�7       ���Y	�h�T���A�
*)

cross_entropy��?


accuracy_1333?1;�U7       ���Y	���T���A�
*)

cross_entropyaD?


accuracy_1�p=?��f
7       ���Y	��U���A�
*)

cross_entropy�?


accuracy_1�p=?�i�	7       ���Y	r�4U���A�
*)

cross_entropy
"?


accuracy_1R�?�'�*7       ���Y	4�V���A�
*)

cross_entropyeq�>


accuracy_1��Q?�i�7       ���Y	��"W���A�
*)

cross_entropy�[�>


accuracy_1�G?3t�7       ���Y	�OW���A�
*)

cross_entropy��>


accuracy_1��L?:��q7       ���Y	nzW���A�
*)

cross_entropy��?


accuracy_1333?e�MT7       ���Y	<q�W���A�
*)

cross_entropyl�-?


accuracy_1
�#?Pm��7       ���Y	���W���A�
*)

cross_entropy�5+?


accuracy_1��?�))u7       ���Y	��W���A�
*)

cross_entropy�?


accuracy_1��(?5�7       ���Y	��X���A�
*)

cross_entropyH�?


accuracy_1
�#?�N��7       ���Y	�ZGX���A�
*)

cross_entropy_�A


accuracy_1R�?U��7       ���Y	5�pX���A�
*)

cross_entropy�6?


accuracy_1��L?H�)|7       ���Y	�45Z���A�
*)

cross_entropy#�?


accuracy_1�Q8?"�x7       ���Y	�_Z���A�
*)

cross_entropye��>


accuracy_1=
W?�~\�7       ���Y	���Z���A�
*)

cross_entropyo�?


accuracy_1{.?���57       ���Y	Vj�Z���A�
*)

cross_entropy�f ?


accuracy_1=
W?�=��7       ���Y	���Z���A�
*)

cross_entropy�K?


accuracy_1�Q8?z|��7       ���Y	�G[���A�
*)

cross_entropy,a�>


accuracy_1\�B?5М7       ���Y	�S-[���A�
*)

cross_entropy%�	?


accuracy_1{.?3��#7       ���Y	��U[���A�
*)

cross_entropyA�?


accuracy_1�Q8?h�
�7       ���Y	�[���A�
*)

cross_entropyƺ&?


accuracy_1R�?�K��7       ���Y	kQ�[���A�
*)

cross_entropyCY�>


accuracy_1��L?c_�7       ���Y	�r]���A�
*)

cross_entropy��?


accuracy_1
�#?�n��7       ���Y	��]���A�
*)

cross_entropyzK<?


accuracy_1��?���7       ���Y	��]���A�
*)

cross_entropy�V ?


accuracy_1\�B?Ň�7       ���Y	H�]���A�
*)

cross_entropy�"?


accuracy_1��?��g/7       ���Y	�b^���A�
*)

cross_entropy%*#?


accuracy_1
�#?���7       ���Y	EA^���A�
*)

cross_entropy��	?


accuracy_1{.?4߯�7       ���Y	��k^���A�
*)

cross_entropy�<?


accuracy_1�Q8?�� 7       ���Y	L��^���A�
*)

cross_entropyұ
?


accuracy_1�G?.U7       ���Y	vI�^���A�
*)

cross_entropy�1?


accuracy_1333?]d��7       ���Y	���^���A�
*)

cross_entropyLX?


accuracy_1333?��7       ���Y	��`���A�
*)

cross_entropy�e�?


accuracy_1{.?����7       ���Y	��`���A�
*)

cross_entropy\�?


accuracy_1333?��-@7       ���Y	�va���A�
*)

cross_entropy��?


accuracy_1{.?��;H7       ���Y	��/a���A�
*)

cross_entropy�2?


accuracy_1{.?X���7       ���Y	MYa���A�
*)

cross_entropy6
6?


accuracy_1��?	O�:7       ���Y	��a���A�
*)

cross_entropy�P�>


accuracy_1333?Dn7       ���Y	�ήa���A�
*)

cross_entropy&?


accuracy_1{.?��RL7       ���Y	Ŗ�a���A�
*)

cross_entropyx{?


accuracy_1333?pF٠7       ���Y	�Xb���A�
*)

cross_entropy�?


accuracy_1333?�nb7       ���Y	R>.b���A�
*)

cross_entropyj�?


accuracy_1�Q8?_>7       ���Y	��c���A�
*)

cross_entropy"�
?


accuracy_1�Q8?�M*�7       ���Y	�bd���A�
*)

cross_entropyG�?


accuracy_1{.?�87       ���Y	�YHd���A�
*)

cross_entropyE�>


accuracy_1\�B?Z��~7       ���Y	��qd���A�
*)

cross_entropy��>


accuracy_1\�B?B>�7       ���Y	>��d���A�
*)

cross_entropyO�?


accuracy_1�Q8?��f]7       ���Y	�T�d���A�
*)

cross_entropy��?


accuracy_1333?h
�L7       ���Y	�d���A�
*)

cross_entropy&?


accuracy_1{.?��I7       ���Y	Fe���A�
*)

cross_entropyJi�>


accuracy_1��(?�v,7       ���Y	+Fe���A�
*)

cross_entropyJ��>


accuracy_1=
W?X9y<7       ���Y	�Ppe���A�
*)

cross_entropy#T%?


accuracy_1{.?��X�7       ���Y	��2g���A�
*)

cross_entropy!��>


accuracy_1\�B?0�z7       ���Y	��]g���A�
*)

cross_entropy�R?


accuracy_1��Q?Pb"{7       ���Y	9Ήg���A�
*)

cross_entropy�
?


accuracy_1�Q8?���7       ���Y	k3�g���A�
*)

cross_entropyf@:?


accuracy_1333?��j7       ���Y	ȃ�g���A�
*)

cross_entropy_?


accuracy_1��(?(~[�7       ���Y	�^
h���A�
*)

cross_entropy�N
?


accuracy_1\�B?k���7       ���Y	��5h���A�
*)

cross_entropy�m)?


accuracy_1R�?tW��7       ���Y	�_h���A�
*)

cross_entropyc7?


accuracy_1
�#?�7h�7       ���Y	v!�h���A�
*)

cross_entropy�%d?


accuracy_1��?��c7       ���Y	��h���A�
*)

cross_entropy�0?


accuracy_1
�#?<��7       ���Y	쿀j���A�
*)

cross_entropyŶ+?


accuracy_1
�#?�(GM7       ���Y	
A�j���A�
*)

cross_entropyo�>


accuracy_1=
W?�ܩ�7       ���Y	��j���A�
*)

cross_entropyz84?


accuracy_1\�B?�܏�7       ���Y	�k���A�
*)

cross_entropy��?


accuracy_1��(?����7       ���Y	�7+k���A�
*)

cross_entropyS ?


accuracy_1�p=?�/�7       ���Y	j�Sk���A�
*)

cross_entropyv�;?


accuracy_1�z?��#�7       ���Y		�}k���A�
*)

cross_entropym7�>


accuracy_1��L?d�}�7       ���Y	0`�k���A�
*)

cross_entropy�
?


accuracy_1333?5�Kh7       ���Y	���k���A�
*)

cross_entropy���>


accuracy_1�p=?�iӎ7       ���Y	�c�k���A�
*)

cross_entropy8�@?


accuracy_1)\?.e=7       ���Y	>��m���A�
*)

cross_entropy$^?


accuracy_1��?Ɇ�7       ���Y	��m���A�
*)

cross_entropy�q?


accuracy_1333?��qb7       ���Y	��n���A�
*)

cross_entropyfk?


accuracy_1
�#?"�c�7       ���Y	G�En���A�
*)

cross_entropy���>


accuracy_1��Q?l�.7       ���Y	ȭqn���A�
*)

cross_entropy��?


accuracy_1\�B?��F97       ���Y	T��n���A�
*)

cross_entropy���>


accuracy_1�G?��0�7       ���Y	"�n���A�
*)

cross_entropy�W?


accuracy_1��(?.��N7       ���Y	���n���A�
*)

cross_entropy�T?


accuracy_1��Q?��{�7       ���Y	��o���A�*)

cross_entropy&�?


accuracy_1333?'�P7       ���Y	��Ao���A�*)

cross_entropyq�?


accuracy_1333?cwm7       ���Y	$�q���A�*)

cross_entropyS�?


accuracy_1333?
�υ7       ���Y	u�4q���A�*)

cross_entropy�?


accuracy_1�p=?���7       ���Y	?�^q���A�*)

cross_entropyFh?


accuracy_1��L?$o�7       ���Y	qh�q���A�*)

cross_entropy�?


accuracy_1�Q8?�';�7       ���Y	�x�q���A�*)

cross_entropy}l
?


accuracy_1333?̗�B7       ���Y	_�q���A�*)

cross_entropy3�@?


accuracy_1�z?]8��7       ���Y	)�r���A�*)

cross_entropy��>


accuracy_1��Q?���7       ���Y	��-r���A�*)

cross_entropy{�>


accuracy_1\�B?Q��07       ���Y	��Xr���A�*)

cross_entropy�T	?


accuracy_1��(?"Z�l7       ���Y	W��r���A�*)

cross_entropy6M�>


accuracy_1��Q?鼗�7       ���Y	�MJt���A�*)

cross_entropy,??


accuracy_1\�B?̥gi7       ���Y	__rt���A�*)

cross_entropyC��>


accuracy_1�Q8?�cv�7       ���Y	��t���A�*)

cross_entropy��?


accuracy_1{.?,/%�7       ���Y	��t���A�*)

cross_entropylp�>


accuracy_1��Q?qb��7       ���Y	���t���A�*)

cross_entropy�?


accuracy_1��(?	ag�7       ���Y	�Ju���A�*)

cross_entropy�(	?


accuracy_1��(?��7       ���Y	��?u���A�*)

cross_entropy{�?


accuracy_1333?���7       ���Y	a�ju���A�*)

cross_entropyqp)?


accuracy_1
�#?_�eg7       ���Y	{�u���A�*)

cross_entropy�?


accuracy_1{.?��
�7       ���Y	�u���A�*)

cross_entropy�G?


accuracy_1333?_i�7       ���Y	��w���A�*)

cross_entropy�?


accuracy_1\�B?nb�C7       ���Y	6�w���A�*)

cross_entropy���>


accuracy_1\�B?۾�O7       ���Y	�H�w���A�*)

cross_entropy�(?


accuracy_1��(?.8�7       ���Y	s�x���A�*)

cross_entropy&$/?


accuracy_1��(?|��-7       ���Y	�E.x���A�*)

cross_entropy���>


accuracy_1\�B?g��7       ���Y	sPYx���A�*)

cross_entropy\1'?


accuracy_1��(?���7       ���Y	EJ�x���A�*)

cross_entropy�Y?


accuracy_1333?�'{�7       ���Y	��x���A�*)

cross_entropym�?


accuracy_1
�#?�*�7       ���Y	]i�x���A�*)

cross_entropy�
?


accuracy_1�G?
ǌ7       ���Y	���x���A�*)

cross_entropy�+?


accuracy_1333?���W7       ���Y	C�z���A�*)

cross_entropyþ)?


accuracy_1
�#?Iд�7       ���Y	/C�z���A�*)

cross_entropy�o?


accuracy_1��?���7       ���Y	ܟ{���A�*)

cross_entropyt�?


accuracy_1
�#?܄9�7       ���Y	��G{���A�*)

cross_entropy!#	?


accuracy_1�p=?҄�7       ���Y	߬q{���A�*)

cross_entropy�%�>


accuracy_1�p=?��}�7       ���Y	Z�{���A�*)

cross_entropy_� ?


accuracy_1�p=?&lh47       ���Y	�
�{���A�*)

cross_entropy�?


accuracy_1{.?�#�7       ���Y	TI�{���A�*)

cross_entropy��?


accuracy_1��(?���7       ���Y	g�|���A�*)

cross_entropy�	?


accuracy_1{.?w��7       ���Y	��D|���A�*)

cross_entropy��?


accuracy_1�Q8?R��77       ���Y	>~���A�*)

cross_entropy]�?


accuracy_1\�B?x�T7       ���Y	�q6~���A�*)

cross_entropy���>


accuracy_1\�B?��7       ���Y	�,c~���A�*)

cross_entropy�?


accuracy_1�Q8?Bj��7       ���Y	2O�~���A�*)

cross_entropy�w ?


accuracy_1��(?�%��7       ���Y	+J�~���A�*)

cross_entropyd�?


accuracy_1��Q?)i}7       ���Y	��~���A�*)

cross_entropy�?


accuracy_1{.?x��x7       ���Y		���A�*)

cross_entropy;[?


accuracy_1�G?;�d�7       ���Y	�E6���A�*)

cross_entropy�j?


accuracy_1R�?	<<I7       ���Y	�h`���A�*)

cross_entropy3�%?


accuracy_1��(?��V�7       ���Y	�R����A�*)

cross_entropy��?


accuracy_1{.?���7       ���Y	?AW����A�*)

cross_entropy:�?


accuracy_1��Q?��)7       ���Y	�6�����A�*)

cross_entropy�H?


accuracy_1�G?o5*7       ���Y	[���A�*)

cross_entropy1i�>


accuracy_1=
W?��47       ���Y	�-ԁ���A�*)

cross_entropy=P?


accuracy_1�Q8?�b[^7       ���Y	-������A�*)

cross_entropy�[?


accuracy_1�p=?���:7       ���Y	��'����A�*)

cross_entropy��>


accuracy_1�p=?1��|7       ���Y	��S����A�*)

cross_entropy���>


accuracy_1�G?��7       ���Y	w�����A�*)

cross_entropy��?


accuracy_1333?0m��7       ���Y	�z�����A�*)

cross_entropy�U?


accuracy_1\�B?`�7       ���Y	��Ԃ���A�*)

cross_entropy5�?


accuracy_1333?����7       ���Y	S!�����A�*)

cross_entropy�C?


accuracy_1R�?z��7       ���Y	�����A�*)

cross_entropy�?


accuracy_1333?p@�7       ���Y	�������A�*)

cross_entropy��?


accuracy_1{.?1F��7       ���Y	�g����A�*)

cross_entropyƾ?


accuracy_1�Q8?����7       ���Y	G[H����A�*)

cross_entropy��?


accuracy_1{.?LE�@7       ���Y	�(q����A�*)

cross_entropys~?


accuracy_1R�?���7       ���Y	"�����A�*)

cross_entropy�&7?


accuracy_1333?p�C37       ���Y	�Ņ���A�*)

cross_entropy,.?


accuracy_1�G?YU�7       ���Y	�4템��A�*)

cross_entropy�e?


accuracy_1{.?]�!7       ���Y	 ����A�*)

cross_entropyo[�>


accuracy_1=
W?�G�7       ���Y	Q�އ���A�*)

cross_entropyR�	?


accuracy_1{.?�8�D7       ���Y	3�����A�*)

cross_entropyF
?


accuracy_1{.? 2�a7       ���Y	2����A�*)

cross_entropy��?


accuracy_1
�#?ϰ��7       ���Y	c�\����A�*)

cross_entropy�)?


accuracy_1�Q8?JW(>7       ���Y	#������A�*)

cross_entropy�?


accuracy_1\�B?1�7       ���Y	Ұ����A�*)

cross_entropy��?


accuracy_1
�#?F��7       ���Y	�+ۈ���A�*)

cross_entropyf?


accuracy_1
�#?�e;�7       ���Y	U5����A�*)

cross_entropy� ?


accuracy_1��L?��-�7       ���Y	�.����A�*)

cross_entropy�~#?


accuracy_1�Q8?�;�77       ���Y	w�Y����A�*)

cross_entropy���>


accuracy_1��L?\�٘7       ���Y	L����A�*)

cross_entropy��"?


accuracy_1��?�N