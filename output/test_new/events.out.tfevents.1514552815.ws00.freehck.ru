       �K"	  �{���Abrain.Event:2�u���      �*�	^�{���A"ݗ
l
xPlaceholder*
dtype0*+
_output_shapes
:���������d* 
shape:���������d
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      d   *
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
seed2 *
T0*
dtype0*&
_output_shapes
:dd*

seed 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dd
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dd
�
Variable
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable*
validate_shape(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
R
ConstConst*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_1*
validate_shape(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
Conv2DConv2DReshapeVariable/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"   d      d   *
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
seed2 *
T0*
dtype0*&
_output_shapes
:dd*

seed 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
:dd
�

Variable_2
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2*
validate_shape(
w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
T
Const_1Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_3
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_3*
validate_shape(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
Conv2D_1Conv2DReshapeVariable_2/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   d      d   *
_output_shapes
:
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
seed2 *
T0*
dtype0*&
_output_shapes
:dd*

seed 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
:dd
�

Variable_4
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
T
Const_2Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_5
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_5*
validate_shape(
k
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
Conv2D_2Conv2DReshapeVariable_4/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
N*
T0*

Tidx0*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
dtype0*
valueB"����,  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
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
seed2 *
T0*
dtype0*(
_output_shapes
:����������*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
O
dropout/divRealDiv	Reshape_1	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB",     *
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
seed2 *
T0*
dtype0*
_output_shapes
:	�*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6*
validate_shape(
p
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_7
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:
�
Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@Variable_7*
validate_shape(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
MatMulMatMuldropout/mulVariable_6/read*
transpose_b( *
transpose_a( *'
_output_shapes
:���������*
T0
U
addAddMatMulVariable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
H
ShapeShapeadd*
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
J
Shape_1Shapeadd*
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
Slice/beginPackSub*
N*
T0*

axis *
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
Slice/size*
Index0*
T0*
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
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
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
Slice_1/beginPackSub_1*
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
_output_shapes
:
d
concat_2/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
O
concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
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
Slice_2/sizePackSub_2*
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
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
MeanMean	Reshape_4Const_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
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
gradients/Mean_grad/ConstConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
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
T0*'
_output_shapes
:���������*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
a
gradients/Reshape_2_grad/ShapeShapeadd*
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
^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *(
_output_shapes
:����������*
T0
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	�*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	�
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
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
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
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:����������
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
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
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
d
gradients/Reshape_1_grad/ShapeShapeconcat*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:����������
\
gradients/concat_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
b
gradients/concat_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
N*
T0*
out_type0*&
_output_shapes
:::
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2*/
_output_shapes
:���������d
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
strides
*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:���������d
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������d
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������d
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:d
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������d
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:d
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
N*
T0*
out_type0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N*
T0*
out_type0* 
_output_shapes
::
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@Variable*
validate_shape(
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@Variable*
validate_shape(
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable*
validate_shape(
{
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable*
validate_shape(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_1/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_1*
validate_shape(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_1*
validate_shape(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_2/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2*
validate_shape(
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_2/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2*
validate_shape(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_3/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_3*
validate_shape(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_3/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_3*
validate_shape(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_4/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4*
validate_shape(
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_4/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4*
validate_shape(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_5/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_5*
validate_shape(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_5/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:d*
_class
loc:@Variable_5*
validate_shape(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�*
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6*
validate_shape(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�*
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6*
validate_shape(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:
�
Variable_7/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:*
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@Variable_7*
validate_shape(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:*
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*
_class
loc:@Variable_7*
validate_shape(
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
T0*&
_output_shapes
:dd*
_class
loc:@Variable*
use_nesterov( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:d*
_class
loc:@Variable_1*
use_nesterov( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2*
use_nesterov( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:d*
_class
loc:@Variable_3*
use_nesterov( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4*
use_nesterov( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:d*
_class
loc:@Variable_5*
use_nesterov( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:	�*
_class
loc:@Variable_6*
use_nesterov( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes
:*
_class
loc:@Variable_7*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@Variable*
validate_shape(
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
Adam/mul_1*
use_locking( *
T0*
_output_shapes
: *
_class
loc:@Variable*
validate_shape(
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*

Tidx0*#
_output_shapes
:���������*
T0
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:���������*
T0
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
accuracyMeanCast_1Const_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
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
: "�����      eef�	W��{���AJ��
�'�'
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
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514ݗ
l
xPlaceholder*
dtype0*+
_output_shapes
:���������d* 
shape:���������d
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*
dtype0*%
valueB"   d      d   *
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
dtype0*
T0*
seed2 *&
_output_shapes
:dd*

seed 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dd
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dd
�
Variable
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dd*
validate_shape(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
R
ConstConst*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:d*
validate_shape(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
Conv2DConv2DReshapeVariable/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"   d      d   *
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
dtype0*
T0*
seed2 *&
_output_shapes
:dd*

seed 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
:dd
�

Variable_2
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd*
validate_shape(
w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
T
Const_1Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_3
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
Conv2D_1Conv2DReshapeVariable_2/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   d      d   *
_output_shapes
:
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
dtype0*
T0*
seed2 *&
_output_shapes
:dd*

seed 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
:dd
�

Variable_4
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
T
Const_2Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_5
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d
�
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(
k
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
Conv2D_2Conv2DReshapeVariable_4/read*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
N*
T0*

Tidx0*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
dtype0*
valueB"����,  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
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
dtype0*
T0*
seed2 *(
_output_shapes
:����������*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
O
dropout/divRealDiv	Reshape_1	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB",     *
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
dtype0*
T0*
seed2 *
_output_shapes
:	�*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�*
validate_shape(
p
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*���=*
_output_shapes
:
v

Variable_7
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:
�
Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
MatMulMatMuldropout/mulVariable_6/read*
transpose_a( *'
_output_shapes
:���������*
T0*
transpose_b( 
U
addAddMatMulVariable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
H
ShapeShapeadd*
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
J
Shape_1Shapeadd*
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
Slice/beginPackSub*
N*
T0*

axis *
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
Slice/size*
Index0*
T0*
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
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
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
Slice_1/beginPackSub_1*
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
_output_shapes
:
d
concat_2/values_0Const*
dtype0*
valueB:
���������*
_output_shapes
:
O
concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
N*
T0*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
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
Slice_2/sizePackSub_2*
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
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
MeanMean	Reshape_4Const_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
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
gradients/Mean_grad/ConstConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
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
T0*'
_output_shapes
:���������*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
a
gradients/Reshape_2_grad/ShapeShapeadd*
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
^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *(
_output_shapes
:����������*
T0*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	�*
T0*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	�
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
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
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
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:����������
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
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
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
d
gradients/Reshape_1_grad/ShapeShapeconcat*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:����������
\
gradients/concat_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
b
gradients/concat_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
N*
T0*
out_type0*&
_output_shapes
:::
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*J
_output_shapes8
6:4������������������������������������
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2*/
_output_shapes
:���������d
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
strides
*
ksize
*
data_formatNHWC*
T0*
paddingVALID*/
_output_shapes
:���������d
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:���������d
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������d
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������d
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:d
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������d
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:d
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
N*
T0*
out_type0* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N*
T0*
out_type0* 
_output_shapes
::
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
T0*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *fff?*
_output_shapes
: 
�
beta1_power
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *w�?*
_output_shapes
: 
�
beta2_power
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dd*
validate_shape(
{
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*&
_output_shapes
:dd*
validate_shape(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_1/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:d*
validate_shape(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_1/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
_output_shapes
:d*
validate_shape(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_2/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd*
validate_shape(
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_2/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd*
validate_shape(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_3/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_3/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_4/Adam
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd*
validate_shape(
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
�
Variable_4/Adam_1
VariableV2*
	container *
shared_name *
dtype0*&
_output_shapes
:dd*
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd*
validate_shape(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_5/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    *
_output_shapes
:d
�
Variable_5/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:d*
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�*
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�*
validate_shape(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
Variable_6/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	�*
shape:	�*
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�*
validate_shape(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:
�
Variable_7/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:*
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:*
shape:*
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(
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
loc:@Variable*&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
_output_shapes
:	�*
use_nesterov( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
_output_shapes
:*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(
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
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*

Tidx0*#
_output_shapes
:���������*
T0
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:���������*
T0
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
accuracyMeanCast_1Const_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
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
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0"�
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
Adam".
	summaries!

cross_entropy:0
accuracy_1:0��4       ^3\	��{���A*)

cross_entropy�T�@


accuracy_1{�>���|6       OW��	�|���A(*)

cross_entropy&��?


accuracy_1{?
u�6       OW��	(8|���AP*)

cross_entropy��?


accuracy_1��?%��6       OW��	�SX|���Ax*)

cross_entropy(r�?


accuracy_15^?����7       ���Y	C�y|���A�*)

cross_entropy�w?


accuracy_1�?y9`�7       ���Y	tB�|���A�*)

cross_entropyXXt?


accuracy_1��?r#�7       ���Y	�s�|���A�*)

cross_entropy��o?


accuracy_1�|?֚�t7       ���Y	��|���A�*)

cross_entropy��f?


accuracy_1
�#?�j��7       ���Y	�|���A�*)

cross_entropy�)a?


accuracy_1ˡ%?nq�|7       ���Y	 �}���A�*)

cross_entropyx�]?


accuracy_1ff&?iy�7       ���Y	l�7}���A�*)

cross_entropy��[?


accuracy_1�&?�WB7       ���Y	�W}���A�*)

cross_entropy�X?


accuracy_1�'?>�M7       ���Y	o�v}���A�*)

cross_entropy_�V?


accuracy_1��(?�O�7       ���Y	�o�}���A�*)

cross_entropyًT?


accuracy_1�~*?ΟB,7       ���Y	�P�}���A�*)

cross_entropy��R?


accuracy_1�~*?���17       ���Y	e��}���A�*)

cross_entropy��Q?


accuracy_1��*?�FB�7       ���Y	�M~���A�*)

cross_entropy�IS?


accuracy_1�+?ߍ�57       ���Y	�,(~���A�*)

cross_entropy �P?


accuracy_1�I,?�c�7       ���Y	��G~���A�*)

cross_entropy�N?


accuracy_1h�-?L�
V7       ���Y	�jg~���A�*)

cross_entropy��M?


accuracy_1�O-?R#�G7       ���Y	��~���A�*)

cross_entropy��K?


accuracy_1�O-?uO7E7       ���Y	��~���A�*)

cross_entropy�>Y?


accuracy_1��.?ߏL�7       ���Y	�9�~���A�*)

cross_entropy-`K?


accuracy_1)\/?�&{7       ���Y	L!�~���A�*)

cross_entropy��I?


accuracy_1�/?NgO7       ���Y	 ���A�*)

cross_entropy�*K?


accuracy_1)\/?��37       ���Y	�f$���A�*)

cross_entropy��G?


accuracy_1`�0?���7       ���Y	�$W���A�*)

cross_entropy_�G?


accuracy_1��1?���7       ���Y	!w���A�*)

cross_entropy�iF?


accuracy_1�n2?1D�7       ���Y	g2����A�*)

cross_entropy(DF?


accuracy_1��1?/q��7       ���Y	z����A�	*)

cross_entropy��D?


accuracy_1��1?�o�v7       ���Y	!�����A�	*)

cross_entropy��C?


accuracy_1��2?�)�7       ���Y	U5����A�	*)

cross_entropy��C?


accuracy_1��2?e}�s7       ���Y	;n����A�
*)

cross_entropy��C?


accuracy_1��3?pr�7       ���Y	<D6����A�
*)

cross_entropy++C?


accuracy_1��3?�xN�7       ���Y	ȊU����A�
*)

cross_entropyg]B?


accuracy_1X94?GJO_7       ���Y	%Hu����A�
*)

cross_entropyB?


accuracy_1X94?��0�7       ���Y	�r�����A�*)

cross_entropy�.A?


accuracy_1��4?!`��7       ���Y	������A�*)

cross_entropy�B?


accuracy_1-2?���S7       ���Y	t������A�*)

cross_entropy�@?


accuracy_1}?5?���7       ���Y	�����A�*)

cross_entropy�;??


accuracy_1j�4?����7       ���Y	A� ����A�*)

cross_entropy9??


accuracy_1��2?��|7       ���Y	��@����A�*)

cross_entropy7e??


accuracy_1�z4?��7       ���Y	��_����A�*)

cross_entropy�3A?


accuracy_1X94?�T�
7       ���Y	������A�*)

cross_entropy��<?


accuracy_1}?5?80)�7       ���Y	������A�*)

cross_entropyV�=?


accuracy_1��5?WΥ�7       ���Y	ⶾ����A�*)

cross_entropy�r<?


accuracy_1��4?�>��7       ���Y	�8ށ���A�*)

cross_entropy��;?


accuracy_1��4?�|��7       ���Y	�
�����A�*)

cross_entropy��;?


accuracy_1�E6?�UI�7       ���Y	�1����A�*)

cross_entropyl;?


accuracy_1��5??1w7       ���Y	,3=����A�*)

cross_entropy5|;?


accuracy_1}?5?�x�m7       ���Y	�\����A�*)

cross_entropy��<?


accuracy_1��5?a�
}7       ���Y	�f�����A�*)

cross_entropy �:?


accuracy_16?u�Y�7       ���Y	Ol�����A�*)

cross_entropy�:?


accuracy_1�K7?!f��7       ���Y	��͂���A�*)

cross_entropy��9?


accuracy_1+�6?�,c�7       ���Y	��킏��A�*)

cross_entropy��9?


accuracy_1=
7?mB�7       ���Y	������A�*)

cross_entropy�8?


accuracy_1+�6?�O�7       ���Y	i�,����A�*)

cross_entropy2�7?


accuracy_1+�6?~[�7       ���Y	{L����A�*)

cross_entropy��7?


accuracy_1��6?jHC7       ���Y	��k����A�*)

cross_entropy�7?


accuracy_1=
7?��U7       ���Y	禋����A�*)

cross_entropy!7?


accuracy_1�E6?Ԟ1�7       ���Y	� �����A�*)

cross_entropy�6?


accuracy_16?z��7       ���Y	b
˃���A�*)

cross_entropy�5?


accuracy_1��6?,J�E7       ���Y	7Eꃏ��A�*)

cross_entropy��5?


accuracy_1�K7?�8�7       ���Y	�����A�*)

cross_entropy=25?


accuracy_1��7?i���7       ���Y	
�=����A�*)

cross_entropy�H4?


accuracy_1b8?�i�7       ���Y	{]����A�*)

cross_entropyuK4?


accuracy_1��9?2���7       ���Y	�|����A�*)

cross_entropym4?


accuracy_1u�8?���7       ���Y	������A�*)

cross_entropyf�3?


accuracy_1�9?��L|7       ���Y	�������A�*)

cross_entropy=	4?


accuracy_1u�8?���37       ���Y	�Mۄ���A�*)

cross_entropy�i3?


accuracy_1u�8?#��37       ���Y	�������A�*)

cross_entropyr�2?


accuracy_1#�9?`Uw�7       ���Y	o&����A�*)

cross_entropy�2?


accuracy_1u�8?5��o7       ���Y	�8:����A�*)

cross_entropy"2?


accuracy_1Zd;?�_7       ���Y	�Y����A�*)

cross_entropy�1?


accuracy_1�Q8?�"97       ���Y	�'y����A�*)

cross_entropyR�0?


accuracy_1m�;?0��7       ���Y	������A�*)

cross_entropyZw1?


accuracy_1+�6?e_(7       ���Y	��̅���A�*)

cross_entropy��0?


accuracy_1Zd;?�*��7       ���Y	%O텏��A�*)

cross_entropy�0?


accuracy_1#�9?)�7       ���Y	8�����A�*)

cross_entropy}�/?


accuracy_1�E6?\�+7       ���Y	�X,����A�*)

cross_entropy5i/?


accuracy_1Zd;?�6�Z7       ���Y	<EL����A�*)

cross_entropyJ�/?


accuracy_1��5?؇�7       ���Y	*un����A�*)

cross_entropy^8/?


accuracy_1#�9?9O�s7       ���Y	������A�*)

cross_entropy��/?


accuracy_1+�6?s't�7       ���Y	DW�����A�*)

cross_entropy&�.?


accuracy_1�Q8?9�.K7       ���Y	W�̆���A�*)

cross_entropyDb/?


accuracy_1+�6?�D)17       ���Y	�,솏��A�*)

cross_entropy>.?


accuracy_1�9?uŸ7       ���Y	ҫ����A�*)

cross_entropy�4.?


accuracy_1�K7?;B�G7       ���Y	�+����A�*)

cross_entropyT,-?


accuracy_1u�8?��|7       ���Y	��W����A�*)

cross_entropy|,?


accuracy_15^:?_Fas7       ���Y	p�w����A�*)

cross_entropy��,?


accuracy_1��7?�t�7       ���Y	6(�����A�*)

cross_entropy',?


accuracy_1�:?$��7       ���Y	�������A�*)

cross_entropy�,?


accuracy_1b8?�+T17       ���Y	�\և���A�*)

cross_entropy�,?


accuracy_1��7?�s��7       ���Y		[�����A�*)

cross_entropy�",?


accuracy_1��7?�2r7       ���Y	������A�*)

cross_entropyo�+?


accuracy_1��8?�'�7       ���Y	�g5����A�*)

cross_entropy{�+?


accuracy_1��7?���7       ���Y	��T����A�*)

cross_entropy��*?


accuracy_1�:?�wC�7       ���Y	-[t����A�*)

cross_entropyi�*?


accuracy_1�:?:<-7       ���Y	BP�����A�*)

cross_entropyvU+?


accuracy_1��6?�6O;7       ���Y	������A�*)

cross_entropy��*?


accuracy_1�9?�O�7       ���Y	�҈���A�*)

cross_entropy �*?


accuracy_1�E6?���7       ���Y	�N����A�*)

cross_entropyp*?


accuracy_1�9?M��@7       ���Y	�8$����A�*)

cross_entropy�Q+?


accuracy_1�Q8?+W^	7       ���Y	�D����A� *)

cross_entropyh,?


accuracy_1j<?���7       ���Y	��c����A� *)

cross_entropyK^*?


accuracy_1u�8?;���7       ���Y	�������A� *)

cross_entropy<�(?


accuracy_15^:?>Ԋ�7       ���Y	\͢����A�!*)

cross_entropy6q(?


accuracy_1Zd;?��7       ���Y	�\É���A�!*)

cross_entropy��(?


accuracy_15^:?��Z�7       ���Y	E≏��A�!*)

cross_entropy;=(?


accuracy_1��9?��%7       ���Y	������A�"*)

cross_entropy�'?


accuracy_1�";?�w�7       ���Y	J�!����A�"*)

cross_entropy�'?


accuracy_1H�:?���7       ���Y	n�A����A�"*)

cross_entropyȝ(?


accuracy_1��7?��l7       ���Y	��`����A�#*)

cross_entropy|�'?


accuracy_1��:?�.07       ���Y	�������A�#*)

cross_entropy]m'?


accuracy_1�Q8?�@7       ���Y	�!�����A�#*)

cross_entropy?f(?


accuracy_1�<?Bױ�7       ���Y	5�Ҋ���A�#*)

cross_entropyV'?


accuracy_1�";?߮i7       ���Y	�[󊏑�A�$*)

cross_entropy�r'?


accuracy_1��7?UP0�7       ���Y	0`����A�$*)

cross_entropyU'?


accuracy_1�:?�]�7       ���Y	j^2����A�$*)

cross_entropyZ�(?


accuracy_1P�7?��7       ���Y	�GQ����A�%*)

cross_entropy |&?


accuracy_1Zd;?�B��7       ���Y	��q����A�%*)

cross_entropy�o&?


accuracy_1��8?ap�o7       ���Y	ʐ����A�%*)

cross_entropy&?


accuracy_1�;?�e7       ���Y	�������A�&*)

cross_entropyV�%?


accuracy_15^:?ڄ�7       ���Y	Ћ���A�&*)

cross_entropy�;%?


accuracy_1�(<?�ܲ7       ���Y	���A�&*)

cross_entropy%?


accuracy_1�<?V"`7       ���Y	Os����A�'*)

cross_entropy�%?


accuracy_1��:?�`7       ���Y	��C����A�'*)

cross_entropy��$?


accuracy_1�(<?���7       ���Y	�Tb����A�'*)

cross_entropyˍ%?


accuracy_15^:?){47       ���Y	yZ�����A�(*)

cross_entropy�$?


accuracy_1�<?��v7       ���Y	������A�(*)

cross_entropyA�$?


accuracy_1�;?���57       ���Y	�������A�(*)

cross_entropy!?$?


accuracy_1�<?]�i7       ���Y	Pጏ��A�(*)

cross_entropy��#?


accuracy_1j<?Ol�+7       ���Y	T����A�)*)

cross_entropy�#?


accuracy_1��<?���7       ���Y	a!����A�)*)

cross_entropy�#?


accuracy_1�(<?�J܉7       ���Y	�-A����A�)*)

cross_entropyU#?


accuracy_1j<?{��7       ���Y	��`����A�**)

cross_entropyʷ"?


accuracy_1��<?o�pa7       ���Y	�U�����A�**)

cross_entropy�#?


accuracy_1�;?�g�7       ���Y	�������A�**)

cross_entropy�P"?


accuracy_1�(<?�)*�7       ���Y	�j͍���A�+*)

cross_entropyM�"?


accuracy_1�;?U��7       ���Y	�퍏��A�+*)

cross_entropy�"?


accuracy_1�;?6s37       ���Y	������A�+*)

cross_entropyo"?


accuracy_1�<?��|�7       ���Y	�q,����A�,*)

cross_entropy'�!?


accuracy_1/=?K<�-7       ���Y	�L����A�,*)

cross_entropy��!?


accuracy_1�<?�7       ���Y	/l����A�,*)

cross_entropy��!?


accuracy_1�<?e�7       ���Y	 �����A�-*)

cross_entropy��!?


accuracy_1��<?�m�E7       ���Y	������A�-*)

cross_entropy��"?


accuracy_1�p=?��Z�7       ���Y	�ˎ���A�-*)

cross_entropy�!?


accuracy_1�p=?+��B7       ���Y	&�뎏��A�-*)

cross_entropy�+!?


accuracy_1j<?�B%67       ���Y	������A�.*)

cross_entropy3� ?


accuracy_1��>?�9n7       ���Y	Ƞ+����A�.*)

cross_entropy��?


accuracy_1�<?Z9��7       ���Y	v�K����A�.*)

cross_entropy
v?


accuracy_1-�=?�%�7       ���Y	%�����A�/*)

cross_entropy�- ?


accuracy_1�<?��+�7       ���Y	�g�����A�/*)

cross_entropy��?


accuracy_1�<?ث�7       ���Y	�������A�/*)

cross_entropyh�?


accuracy_1�p=?wg%�7       ���Y	5tޏ���A�0*)

cross_entropy�v?


accuracy_1/=?ĸ�7       ���Y	� �����A�0*)

cross_entropy^?


accuracy_1��>?�Xv7       ���Y	�U����A�0*)

cross_entropy� ?


accuracy_1j<?:�H�7       ���Y	m�>����A�1*)

cross_entropy�M?


accuracy_1�p=?�?&�7       ���Y	��^����A�1*)

cross_entropyɖ?


accuracy_1?5>?��'7       ���Y	�)����A�1*)

cross_entropyK?


accuracy_1-�=?R��7       ���Y	40�����A�2*)

cross_entropy��!?


accuracy_1�(<?i�!/7       ���Y	������A�2*)

cross_entropy��?


accuracy_1�v>?W���7       ���Y	CEߐ���A�2*)

cross_entropy�G?


accuracy_1��=?���7       ���Y	������A�2*)

cross_entropy)�?


accuracy_1��<?����7       ���Y	�2����A�3*)

cross_entropyl?


accuracy_1��>?��ѭ7       ���Y	��Q����A�3*)

cross_entropy�	?


accuracy_1��=?g��7       ���Y	˻q����A�3*)

cross_entropy�?


accuracy_1R�>?5B>7       ���Y	=ő����A�4*)

cross_entropyJ�?


accuracy_1��=?����7       ���Y	D*�����A�4*)

cross_entropyX�?


accuracy_1�v>?��:97       ���Y	4ґ���A�4*)

cross_entropyf�?


accuracy_1-�=?��,�7       ���Y	��򑏑�A�5*)

cross_entropy�?


accuracy_1?5>?��]_7       ���Y	�����A�5*)

cross_entropy��?


accuracy_1��=?e�@ 7       ���Y	hm3����A�5*)

cross_entropyb�?


accuracy_1R�>?���7       ���Y	�S����A�6*)

cross_entropym�?


accuracy_1R�>??�S7       ���Y	�Tt����A�6*)

cross_entropy;�?


accuracy_1?5>?(	�7       ���Y	�ܔ����A�6*)

cross_entropy1a?


accuracy_1�v>?,q%7       ���Y	�$Ȓ���A�7*)

cross_entropy?


accuracy_1-�=?Ԏ��7       ���Y	6�璏��A�7*)

cross_entropyRK?


accuracy_1?5>?��
7       ���Y	�����A�7*)

cross_entropy�D?


accuracy_1-�=?����7       ���Y	�(����A�7*)

cross_entropy�?


accuracy_1�v>?�$`�7       ���Y	]6J����A�8*)

cross_entropy��?


accuracy_1�v>?7ȴ7       ���Y	ugj����A�8*)

cross_entropy~?


accuracy_1R�>?�'��7       ���Y	�2�����A�8*)

cross_entropy�:?


accuracy_1d;??Nzn7       ���Y	������A�9*)

cross_entropy9F?


accuracy_1�;?*��@7       ���Y	�̓���A�9*)

cross_entropy�B?


accuracy_1��=?�D�7       ���Y	��쓏��A�9*)

cross_entropy�?


accuracy_1j<?�&ڔ7       ���Y	|�����A�:*)

cross_entropy�?


accuracy_1d;??q�eA7       ���Y	��-����