       �K"	  @����Abrain.Event:2Lu��a�      !/��	�\����A"��
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
: "��� ,�      (��y	��j����AJ��
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
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0&�#�4       ^3\	�����A*)

cross_entropy�E�A


accuracy_1���>%��O6       OW��	�/����A
*)

cross_entropy;;�B


accuracy_1Z?�Z��6       OW��	�x6����A*)

cross_entropy{$�A


accuracy_1��?�3WD6       OW��	A����A*)

cross_entropy�&{A


accuracy_1�n?<<�'6       OW��	]gP����A(*)

cross_entropy�2�A


accuracy_1�z? p�s6       OW��	��u����A2*)

cross_entropy'IA


accuracy_1+�?i�c$6       OW��	6Ɩ����A<*)

cross_entropyV	�A


accuracy_1��	?!�`m6       OW��	�澻���AF*)

cross_entropyLp�A


accuracy_1'1?��hA6       OW��	�뾚��AP*)

cross_entropyy�A


accuracy_1j?�yF�6       OW��	�&��AZ*)

cross_entropy($A


accuracy_1��?��Q�6       OW��	�TXŚ��Ad*)

cross_entropy9keA


accuracy_1#�	?Lѥ{6       OW��	��Ț��An*)

cross_entropy�@


accuracy_1/?U�Q6       OW��	�$�˚��Ax*)

cross_entropyZ��@


accuracy_1�?��p�7       ���Y	�Ϛ��A�*)

cross_entropyX��@


accuracy_1b?)P�7       ���Y	�%UҚ��A�*)

cross_entropy�o�@


accuracy_1q=?P�pe7       ���Y	C}�՚��A�*)

cross_entropyvݫ@


accuracy_1q=?S�j7       ���Y	���ؚ��A�*)

cross_entropy@


accuracy_1X? ��'7       ���Y	W/ܚ��A�*)

cross_entropy�j�@


accuracy_1L7?�n�7       ���Y	�Xߚ��A�*)

cross_entropyI
{@


accuracy_1+?�b��7       ���Y	�N�⚐�A�*)

cross_entropy��Q@


accuracy_1?T���7       ���Y	"��嚐�A�*)

cross_entropy�1.@


accuracy_1��#?�L�7       ���Y	�>#隐�A�*)

cross_entropy��@


accuracy_19�(?��x7       ���Y	=�\욐�A�*)

cross_entropy���?


accuracy_1��-?>��}7       ���Y	C���A�*)

cross_entropy���?


accuracy_1L7)?U�R�7       ���Y	�@���A�*)

cross_entropy���?


accuracy_1��#?#�*�7       ���Y	�����A�*)

cross_entropy���?


accuracy_1��#?�E�7       ���Y	0�T����A�*)

cross_entropy�ԫ?


accuracy_1'1(?Ub�7       ���Y	o�����A�*)

cross_entropy�?�?


accuracy_1�v?g��7       ���Y	������A�*)

cross_entropy;9�?


accuracy_1-"?o��7       ���Y	�����A�*)

cross_entropy}%�?


accuracy_1�z$?�ZEt7       ���Y	�U���A�*)

cross_entropy���?


accuracy_1�I?6/�7       ���Y	7��	���A�*)

cross_entropy���?


accuracy_17�!?l�R�7       ���Y	^����A�*)

cross_entropy'��?


accuracy_1�  ?[�Z7       ���Y	8>$���A�*)

cross_entropy��{?


accuracy_1sh!?���7       ���Y	~dp���A�*)

cross_entropyɅ?


accuracy_1��?�7       ���Y	dm����A�*)

cross_entropyr?


accuracy_1��?���{7       ���Y	�����A�*)

cross_entropy�i?


accuracy_1��!?�BV�7       ���Y	h/���A�*)

cross_entropy#Ni?


accuracy_1V?��7       ���Y	_7p ���A�*)

cross_entropy[!f?


accuracy_1�|?�\��7       ���Y	ʌ�#���A�*)

cross_entropy9A`?


accuracy_1{?R��x7       ���Y	�K�&���A�*)

cross_entropy̕Y?


accuracy_1�?{v��7       ���Y		�7*���A�*)

cross_entropy9P?


accuracy_1o#?\j��7       ���Y	ת|-���A�*)

cross_entropy�[??


accuracy_1b(?��F7       ���Y	p�0���A�*)

cross_entropy':=?


accuracy_19�(?L�9:7       ���Y	��4���A�*)

cross_entropy�I?


accuracy_1-�-?�BUj7       ���Y	 %D7���A�*)

cross_entropyr?


accuracy_1R�.?�Ux7       ���Y	-��:���A�*)

cross_entropyٺ?


accuracy_1#�)?��%7       ���Y	)��=���A�*)

cross_entropy2n#?


accuracy_1H�*?]�e�7       ���Y	>&A���A�*)

cross_entropy��&?


accuracy_1�~*?��B7       ���Y	��PD���A�*)

cross_entropyXF)?


accuracy_1b(?��8=7       ���Y	�эG���A�*)

cross_entropyu�*?


accuracy_1�)?JP�7       ���Y	�:�J���A�*)

cross_entropy��[?


accuracy_1�Q(?N���7       ���Y	T�N���A�*)

cross_entropyr}?


accuracy_1��?>@Z7       ���Y	{OQ���A�*)

cross_entropyݎj?


accuracy_1��!?׌ �7       ���Y	��T���A�*)

cross_entropyb�W?


accuracy_17�!?���T7       ���Y	���W���A�*)

cross_entropyt�:?


accuracy_1�$?��67       ���Y	b�[���A�*)

cross_entropy�7?


accuracy_1��!?�a��7       ���Y	��U^���A�*)

cross_entropy�>?


accuracy_1�&?�
j�7       ���Y	��a���A�*)

cross_entropy�7=?


accuracy_1��!?�N+�7       ���Y	v��d���A�*)

cross_entropy�3?


accuracy_1�E&?A��U7       ���Y	��h���A�*)

cross_entropy�,3?


accuracy_1�l'?�D�7       ���Y	��Vk���A�*)

cross_entropyG??


accuracy_1��!?Cӝl7       ���Y	%֟n���A�*)

cross_entropy�|<?


accuracy_1w�?UHr�7       ���Y	3��q���A�*)

cross_entropyq�,?


accuracy_1=
'?'d��7       ���Y	�eu���A�*)

cross_entropy� ?


accuracy_1�+?W�`7       ���Y	��[x���A�*)

cross_entropy%g?


accuracy_1-�-?�+7       ���Y	�j�{���A�*)

cross_entropyb�?


accuracy_1�|/?�Gwq7       ���Y	� �~���A�*)

cross_entropyj�?


accuracy_1o3?�Y;7       ���Y	�X7����A�*)

cross_entropy�C?


accuracy_1�x)?�K�7       ���Y	�z����A�*)

cross_entropy�k?


accuracy_1`�0?��7       ���Y	O0�����A�*)

cross_entropy��?


accuracy_1  0?���7       ���Y	�'����A�*)

cross_entropy��?


accuracy_1��1?�v��7       ���Y	�<<����A�*)

cross_entropy�?


accuracy_1m�+?v\pq7       ���Y	<È����A�*)

cross_entropy�_?


accuracy_1%1?|l��7       ���Y	M�ɕ���A�*)

cross_entropy�Y?


accuracy_1sh1?����7       ���Y	� ����A�*)

cross_entropy_-?


accuracy_1�&1?��27       ���Y	k�L����A�*)

cross_entropy�8?


accuracy_1��1?��7       ���Y	2�����A�*)

cross_entropy��?


accuracy_1�/?R��$7       ���Y	��ݢ���A�*)

cross_entropyLe?


accuracy_1w�/?�3)�7       ���Y	|�����A�*)

cross_entropy`6?


accuracy_1�"+?��Z�7       ���Y	3b����A�*)

cross_entropy=�?


accuracy_1V-?�]�f7       ���Y	������A�*)

cross_entropy^�?


accuracy_1d;/?h,�m7       ���Y	�������A�*)

cross_entropy ?


accuracy_1m�+?����7       ���Y	�>����A�*)

cross_entropy-�?


accuracy_1�O-?Q`�7       ���Y	!䈶���A�*)

cross_entropy��?


accuracy_1� 0?�H/�7       ���Y	��ƹ���A�*)

cross_entropy5�?


accuracy_1/�4?�A�u7       ���Y	1X	����A�*)

cross_entropy!?


accuracy_1�z4?�}��7       ���Y	2�F����A�*)

cross_entropy�o?


accuracy_1T�5?��7       ���Y	a1�Û��A�*)

cross_entropy��	?


accuracy_1�:?��4�7       ���Y	���ƛ��A�*)

cross_entropym
?


accuracy_1L79?t���7       ���Y	�Eʛ��A�*)

cross_entropy�H?


accuracy_1��2?
��7       ���Y	q�a͛��A�*)

cross_entropy�?


accuracy_1y�6?6MA7       ���Y	Bj�Л��A�*)

cross_entropy7Z?


accuracy_1=
7?��!7       ���Y	�	�ӛ��A�*)

cross_entropyo4?


accuracy_1�M2?���7       ���Y	?%כ��A�*)

cross_entropy�g?


accuracy_1�7?���37       ���Y	^�bڛ��A�*)

cross_entropyd�?


accuracy_1�E6?�~ɋ7       ���Y	�*�ݛ��A�*)

cross_entropy�?


accuracy_1�$6?���$7       ���Y	*������A�*)

cross_entropy�{?


accuracy_1j�4?�%o�7       ���Y		�7䛐�A�*)

cross_entropy?�?


accuracy_1y�6?����7       ���Y	Oz盐�A�*)

cross_entropy\�?


accuracy_1X94?��:�7       ���Y	�9�ꛐ�A�*)

cross_entropy�f?


accuracy_1�&1?3Zr7       ���Y	e��훐�A�*)

cross_entropy�,?


accuracy_1�l'?O1��7       ���Y	uG��A�*)

cross_entropy�?


accuracy_1��2?��!7       ���Y	������A�*)

cross_entropyș?


accuracy_1�K7?5�CW7       ���Y	�Z�����A�*)

cross_entropy��?


accuracy_1��1?��Z�7       ���Y	������A�*)

cross_entropy7o?


accuracy_1��3?E�97       ���Y	J�e����A�*)

cross_entropy|�?


accuracy_1�$6?v��Y7       ���Y	�����A�*)

cross_entropy�	?


accuracy_1j<?�#fd7       ���Y	������A�*)

cross_entropy�5?


accuracy_1ף@?��7       ���Y	W�-���A�*)

cross_entropy�f?


accuracy_1h�=?��}�7       ���Y	��n���A�*)

cross_entropy���>


accuracy_1!�B?PL�7       ���Y	e����A�*)

cross_entropy$b?


accuracy_1��;?��r�7       ���Y	�����A�*)

cross_entropy��?


accuracy_1�<?ֽ��7       ���Y	ۥ:���A�*)

cross_entropy��?


accuracy_1�@?�-a7       ���Y	$�����A�*)

cross_entropyq?


accuracy_1�v>?.!��7       ���Y	�����A�*)

cross_entropy�`?


accuracy_1#�9?隇�7       ���Y	41���A�	*)

cross_entropy�z?


accuracy_1��>?j@��7       ���Y	X�N"���A�	*)

cross_entropy]"?


accuracy_1m�;?`0�u7       ���Y	7s�%���A�	*)

cross_entropy�I?


accuracy_1�;?�s�S7       ���Y	<`�(���A�	*)

cross_entropyh�?


accuracy_1��:?���V7       ���Y	m�),���A�	*)

cross_entropy��?


accuracy_1��:?T%�7       ���Y	�l/���A�	*)

cross_entropyǳ?


accuracy_1�9?J��7       ���Y	l��2���A�	*)

cross_entropy3�?


accuracy_1�(<?۝�47       ���Y	��5���A�	*)

cross_entropy�W?


accuracy_1�G1?&�"7       ���Y	�;9���A�	*)

cross_entropy�?


accuracy_1�$6?��7       ���Y	�T�<���A�	*)

cross_entropyo�?


accuracy_1��8?�6��7       ���Y	�`�?���A�	*)

cross_entropy��?


accuracy_1m�;?��݌7       ���Y	�C���A�	*)

cross_entropy��?


accuracy_1�C;?��U�7       ���Y	4�cF���A�
*)

cross_entropyOO?


accuracy_1-�=?m�P7       ���Y	�{�I���A�
*)

cross_entropy�?


accuracy_1�v>?�-Z�7       ���Y	P�L���A�
*)

cross_entropy%��>


accuracy_1��A?vիs7       ���Y	��>P���A�
*)

cross_entropy�>


accuracy_1�zD?�I8K7       ���Y	H�S���A�
*)

cross_entropy>��>


accuracy_19�H?��17       ���Y	}�V���A�
*)

cross_entropy�Y�>


accuracy_1�G?d�f�7       ���Y	��	Z���A�
*)

cross_entropy&�>


accuracy_1P�G?pC �7       ���Y	ܬI]���A�
*)

cross_entropy���>


accuracy_1��C?��c7       ���Y	n�`���A�
*)

cross_entropy\K�>


accuracy_1��B?NG�57       ���Y	���c���A�
*)

cross_entropyT�>


accuracy_1�A@?�>A�7       ���Y	�
g���A�
*)

cross_entropy��>


accuracy_1��D?!@q�7       ���Y	�Wj���A�
*)

cross_entropyz!�>


accuracy_1ZD?�0ܻ7       ���Y	��m���A�
*)

cross_entropyO$ ?


accuracy_1�<?Zƭ<7       ���Y	���p���A�*)

cross_entropy'Q?


accuracy_1R�>?�}x7       ���Y	3� t���A�*)

cross_entropy�N�>


accuracy_1�v>?h�[7       ���Y	P9\w���A�*)

cross_entropy,�?


accuracy_1��8?<}��7       ���Y	r;�z���A�*)

cross_entropyl?


accuracy_1  @?��!7       ���Y	��}���A�*)

cross_entropyj?


accuracy_1��;?J��7       ���Y	_�-����A�*)

cross_entropy�?


accuracy_1��A?Q��7       ���Y	�6j����A�*)

cross_entropy�L?


accuracy_1�<?qٶ|7       ���Y	�������A�*)

cross_entropy�?


accuracy_1D�<?���7       ���Y	�L�����A�*)

cross_entropy� ?


accuracy_1�v>?��