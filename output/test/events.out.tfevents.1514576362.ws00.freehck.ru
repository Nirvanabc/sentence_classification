       �K"	  �z���Abrain.Event:2�bL��      �Ai	�[�z���A"��
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
xPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
e
y_Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����   d      
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
T0*
Tshape0
o
truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
Z
truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:dd*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:dd*
T0
�
Variable
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
q
Variable/readIdentityVariable*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
T
Const_1Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_1
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_1/AssignAssign
Variable_1Const_1*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
Conv2DConv2DReshapeVariable/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
O
ReluReluBiasAdd*/
_output_shapes
:���������d*
T0
�
MaxPoolMaxPoolRelu*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

q
truncated_normal_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
\
truncated_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:dd*
T0
�

Variable_2
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
T
Const_2Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_3
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_3/AssignAssign
Variable_3Const_2*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
Conv2D_1Conv2DReshapeVariable_2/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*/
_output_shapes
:���������d*
T0
�
	MaxPool_1MaxPoolRelu_1*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

q
truncated_normal_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
\
truncated_normal_2/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*&
_output_shapes
:dd*
T0
�

Variable_4
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
w
Variable_4/readIdentity
Variable_4*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
T
Const_3Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_5
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_5/AssignAssign
Variable_5Const_3*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
Conv2D_2Conv2DReshapeVariable_4/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*/
_output_shapes
:���������d*
T0
�
	MaxPool_2MaxPoolRelu_2*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
N*0
_output_shapes
:����������*
T0*

Tidx0
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
T0*
Tshape0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
T0*
out_type0
_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
_
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*(
_output_shapes
:����������*
dtype0*

seed *
T0*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
O
dropout/divRealDiv	Reshape_1	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*
valueB",     
\
truncated_normal_3/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_3/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	�*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	�*
T0
�

Variable_6
VariableV2*
shared_name *
_output_shapes
:	�*
shape:	�*
dtype0*
	container 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
p
Variable_6/readIdentity
Variable_6*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_7
VariableV2*
shared_name *
_output_shapes
:*
shape:*
dtype0*
	container 
�
Variable_7/AssignAssign
Variable_7Const_4*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
B
L2LossL2LossVariable_6/read*
_output_shapes
: *
T0
:
addAddConstL2Loss*
_output_shapes
: *
T0
D
L2Loss_1L2LossVariable_7/read*
_output_shapes
: *
T0
<
add_1AddaddL2Loss_1*
_output_shapes
: *
T0
�
MatMulMatMuldropout/mulVariable_6/read*'
_output_shapes
:���������*
T0*
transpose_b( *
transpose_a( 
W
add_2AddMatMulVariable_7/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
J
ShapeShapeadd_2*
_output_shapes
:*
T0*
out_type0
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
L
Shape_1Shapeadd_2*
_output_shapes
:*
T0*
out_type0
G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
N*
_output_shapes
:*
T0*

axis 
T

Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
n
	Reshape_2Reshapeadd_2concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
I
Shape_2Shapey_*
_output_shapes
:*
T0*
out_type0
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
N*
_output_shapes
:*
T0*

axis 
V
Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
N*
_output_shapes
:*
T0*

Tidx0
k
	Reshape_3Reshapey_concat_2*0
_output_shapes
:������������������*
T0*
Tshape0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:���������:������������������*
T0
I
Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
U
Slice_2/sizePackSub_2*
N*
_output_shapes
:*
T0*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
T0*
Tshape0
Q
Const_5Const*
_output_shapes
:*
dtype0*
valueB: 
^
MeanMean	Reshape_4Const_5*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �@
9
mulMulmul/xadd_1*
_output_shapes
: *
T0
8
add_3AddMeanmul*
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
_output_shapes
: *
dtype0*
valueB Bcross_entropy
Z
cross_entropyScalarSummarycross_entropy/tagsadd_3*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
]
gradients/add_3_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
_
gradients/add_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Fill*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/Fill,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshape-gradients/add_3_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
[
gradients/mul_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
]
gradients/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
v
gradients/mul_grad/mulMul/gradients/add_3_grad/tuple/control_dependency_1add_1*
_output_shapes
: *
T0
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
x
gradients/mul_grad/mul_1Mulmul/x/gradients/add_3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
]
gradients/add_1_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
_
gradients/add_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
[
gradients/add_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
]
gradients/add_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum-gradients/add_1_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_grad/Sum_1Sum-gradients/add_1_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/L2Loss_1_grad/mulMulVariable_7/read/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/L2Loss_grad/mulMulVariable_6/read-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0
`
gradients/add_2_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
f
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_6/read*(
_output_shapes
:����������*
T0*
transpose_b(*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
_output_shapes
:	�*
T0*
transpose_b( *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
gradients/AddNAddNgradients/L2Loss_1_grad/mul/gradients/add_2_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/L2Loss_1_grad/mul
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:���������*
T0*
out_type0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*#
_output_shapes
:���������*
T0*
out_type0
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
�
gradients/AddN_1AddNgradients/L2Loss_grad/mul0gradients/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	�*
T0*,
_class"
 loc:@gradients/L2Loss_grad/mul
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
_output_shapes
:*
T0*
out_type0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*#
_output_shapes
:���������*
T0*
out_type0
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
T0*
Tshape0
\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
_output_shapes
: *
T0
b
gradients/concat_grad/ShapeShapeMaxPool*
_output_shapes
:*
T0*
out_type0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
N*&
_output_shapes
:::*
T0*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*.
_class$
" loc:@gradients/concat_grad/Slice
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:���������d*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:���������d*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*/
_output_shapes
:���������d*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:d*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
:d*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N* 
_output_shapes
::*
T0*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
N* 
_output_shapes
::*
T0*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N* 
_output_shapes
::*
T0*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
_class
loc:@Variable*
valueB
 *fff?
�
beta1_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
_class
loc:@Variable*
valueB
 *w�?
�
beta2_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable*%
valueBdd*    
�
Variable/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable*
shape:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable*%
valueBdd*    
�
Variable/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable*
shape:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_1*
valueBd*    
�
Variable_1/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_1*
valueBd*    
�
Variable_1/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_2*%
valueBdd*    
�
Variable_2/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_2*
shape:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
�
#Variable_2/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_2*%
valueBdd*    
�
Variable_2/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_2*
shape:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
�
!Variable_3/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_3*
valueBd*    
�
Variable_3/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_3*
shape:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_3*
valueBd*    
�
Variable_3/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_3*
shape:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
!Variable_4/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_4*%
valueBdd*    
�
Variable_4/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_4*
shape:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
�
Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_4*%
valueBdd*    
�
Variable_4/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_4*
shape:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_5*
valueBd*    
�
Variable_5/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_5*
valueBd*    
�
Variable_5/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
!Variable_6/Adam/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_6*
valueB	�*    
�
Variable_6/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	�*
_class
loc:@Variable_6*
shape:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
#Variable_6/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_6*
valueB	�*    
�
Variable_6/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	�*
_class
loc:@Variable_6*
shape:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_7*
valueB*    
�
Variable_7/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_7*
shape:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:*
T0*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_7*
valueB*    
�
Variable_7/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_7*
shape:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_7
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *��8
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable*
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_1*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable_2*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_3*
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable_4*
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_5*
use_locking( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
_output_shapes
:	�*
T0*
use_nesterov( *
_class
loc:@Variable_6*
use_locking( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
_output_shapes
:*
T0*
use_nesterov( *
_class
loc:@Variable_7*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
v
ArgMaxArgMaxadd_2ArgMax/dimension*#
_output_shapes
:���������*
T0*
output_type0	*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*#
_output_shapes
:���������*
T0*
output_type0	*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*#
_output_shapes
:���������*

SrcT0
*

DstT0
Q
Const_6Const*
_output_shapes
:*
dtype0*
valueB: 
_
accuracyMeanCast_1Const_6*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
Z
accuracy_1/tagsConst*
_output_shapes
: *
dtype0*
valueB B
accuracy_1
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
: " �l��
     �={	�i�z���AJ��
�(�(
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
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
1
L2Loss
t"T
output"T"
Ttype:
2
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
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
l
xPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
e
y_Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����   d      
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
T0*
Tshape0
o
truncated_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
Z
truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:dd*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:dd*
T0
�
Variable
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
q
Variable/readIdentityVariable*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
T
Const_1Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_1
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_1/AssignAssign
Variable_1Const_1*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
Conv2DConv2DReshapeVariable/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
O
ReluReluBiasAdd*/
_output_shapes
:���������d*
T0
�
MaxPoolMaxPoolRelu*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

q
truncated_normal_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
\
truncated_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_1/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:dd*
T0
�

Variable_2
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
T
Const_2Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_3
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_3/AssignAssign
Variable_3Const_2*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
Conv2D_1Conv2DReshapeVariable_2/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
S
Relu_1Relu	BiasAdd_1*/
_output_shapes
:���������d*
T0
�
	MaxPool_1MaxPoolRelu_1*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

q
truncated_normal_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
\
truncated_normal_2/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*&
_output_shapes
:dd*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*&
_output_shapes
:dd*
T0
�

Variable_4
VariableV2*
shared_name *&
_output_shapes
:dd*
shape:dd*
dtype0*
	container 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
w
Variable_4/readIdentity
Variable_4*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
T
Const_3Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_5
VariableV2*
shared_name *
_output_shapes
:d*
shape:d*
dtype0*
	container 
�
Variable_5/AssignAssign
Variable_5Const_3*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
Conv2D_2Conv2DReshapeVariable_4/read*/
_output_shapes
:���������d*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*/
_output_shapes
:���������d*
T0*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*/
_output_shapes
:���������d*
T0
�
	MaxPool_2MaxPoolRelu_2*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
N*0
_output_shapes
:����������*
T0*

Tidx0
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"����,  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
T0*
Tshape0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
T0*
out_type0
_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
_
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*(
_output_shapes
:����������*
dtype0*

seed *
T0*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
O
dropout/divRealDiv	Reshape_1	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*
valueB",     
\
truncated_normal_3/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_3/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	�*
dtype0*

seed *
T0*
seed2 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	�*
T0
�

Variable_6
VariableV2*
shared_name *
_output_shapes
:	�*
shape:	�*
dtype0*
	container 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
p
Variable_6/readIdentity
Variable_6*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_7
VariableV2*
shared_name *
_output_shapes
:*
shape:*
dtype0*
	container 
�
Variable_7/AssignAssign
Variable_7Const_4*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
B
L2LossL2LossVariable_6/read*
_output_shapes
: *
T0
:
addAddConstL2Loss*
_output_shapes
: *
T0
D
L2Loss_1L2LossVariable_7/read*
_output_shapes
: *
T0
<
add_1AddaddL2Loss_1*
_output_shapes
: *
T0
�
MatMulMatMuldropout/mulVariable_6/read*'
_output_shapes
:���������*
T0*
transpose_b( *
transpose_a( 
W
add_2AddMatMulVariable_7/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
J
ShapeShapeadd_2*
_output_shapes
:*
T0*
out_type0
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
L
Shape_1Shapeadd_2*
_output_shapes
:*
T0*
out_type0
G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
N*
_output_shapes
:*
T0*

axis 
T

Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
n
	Reshape_2Reshapeadd_2concat_1*0
_output_shapes
:������������������*
T0*
Tshape0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
I
Shape_2Shapey_*
_output_shapes
:*
T0*
out_type0
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
N*
_output_shapes
:*
T0*

axis 
V
Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������
O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
N*
_output_shapes
:*
T0*

Tidx0
k
	Reshape_3Reshapey_concat_2*0
_output_shapes
:������������������*
T0*
Tshape0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:���������:������������������*
T0
I
Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
U
Slice_2/sizePackSub_2*
N*
_output_shapes
:*
T0*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
T0*
Tshape0
Q
Const_5Const*
_output_shapes
:*
dtype0*
valueB: 
^
MeanMean	Reshape_4Const_5*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �@
9
mulMulmul/xadd_1*
_output_shapes
: *
T0
8
add_3AddMeanmul*
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
_output_shapes
: *
dtype0*
valueB Bcross_entropy
Z
cross_entropyScalarSummarycross_entropy/tagsadd_3*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
]
gradients/add_3_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
_
gradients/add_3_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Fill*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/Fill,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshape-gradients/add_3_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
[
gradients/mul_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
]
gradients/mul_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
v
gradients/mul_grad/mulMul/gradients/add_3_grad/tuple/control_dependency_1add_1*
_output_shapes
: *
T0
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
x
gradients/mul_grad/mul_1Mulmul/x/gradients/add_3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
]
gradients/add_1_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
_
gradients/add_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
[
gradients/add_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
]
gradients/add_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum-gradients/add_1_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/add_grad/Sum_1Sum-gradients/add_1_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/L2Loss_1_grad/mulMulVariable_7/read/gradients/add_1_grad/tuple/control_dependency_1*
_output_shapes
:*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_2*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/L2Loss_grad/mulMulVariable_6/read-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0
`
gradients/add_2_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
f
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_2_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_6/read*(
_output_shapes
:����������*
T0*
transpose_b(*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul-gradients/add_2_grad/tuple/control_dependency*
_output_shapes
:	�*
T0*
transpose_b( *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
�
gradients/AddNAddNgradients/L2Loss_1_grad/mul/gradients/add_2_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/L2Loss_1_grad/mul
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:���������*
T0*
out_type0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*#
_output_shapes
:���������*
T0*
out_type0
�
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
�
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
�
gradients/AddN_1AddNgradients/L2Loss_grad/mul0gradients/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	�*
T0*,
_class"
 loc:@gradients/L2Loss_grad/mul
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
_output_shapes
:*
T0*
out_type0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*#
_output_shapes
:���������*
T0*
out_type0
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
_output_shapes
:*
T0*
out_type0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
T0*
Tshape0
\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
_output_shapes
: *
T0
b
gradients/concat_grad/ShapeShapeMaxPool*
_output_shapes
:*
T0*
out_type0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
N*&
_output_shapes
:::*
T0*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*.
_class$
" loc:@gradients/concat_grad/Slice
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*/
_output_shapes
:���������d*
T0*
paddingVALID*
data_formatNHWC*
strides

�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:���������d*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:���������d*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*/
_output_shapes
:���������d*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:d*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
_output_shapes
:d*
T0*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
:d*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N* 
_output_shapes
::*
T0*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
N* 
_output_shapes
::*
T0*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N* 
_output_shapes
::*
T0*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������d*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
:dd*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
_class
loc:@Variable*
valueB
 *fff?
�
beta1_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
_class
loc:@Variable*
valueB
 *w�?
�
beta2_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
shape: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable*%
valueBdd*    
�
Variable/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable*
shape:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable*%
valueBdd*    
�
Variable/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable*
shape:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_1*
valueBd*    
�
Variable_1/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_1*
valueBd*    
�
Variable_1/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_2*%
valueBdd*    
�
Variable_2/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_2*
shape:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
�
#Variable_2/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_2*%
valueBdd*    
�
Variable_2/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_2*
shape:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_2
�
!Variable_3/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_3*
valueBd*    
�
Variable_3/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_3*
shape:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_3*
valueBd*    
�
Variable_3/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_3*
shape:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_3
�
!Variable_4/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_4*%
valueBdd*    
�
Variable_4/Adam
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_4*
shape:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
�
Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_4*%
valueBdd*    
�
Variable_4/Adam_1
VariableV2*
dtype0*
	container *
shared_name *&
_output_shapes
:dd*
_class
loc:@Variable_4*
shape:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*&
_output_shapes
:dd*
T0*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_5*
valueBd*    
�
Variable_5/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_5*
valueBd*    
�
Variable_5/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_output_shapes
:d*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
�
!Variable_6/Adam/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_6*
valueB	�*    
�
Variable_6/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	�*
_class
loc:@Variable_6*
shape:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
#Variable_6/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_6*
valueB	�*    
�
Variable_6/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	�*
_class
loc:@Variable_6*
shape:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes
:	�*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes
:	�*
T0*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_7*
valueB*    
�
Variable_7/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_7*
shape:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:*
T0*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_7*
valueB*    
�
Variable_7/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_7*
shape:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_7
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *��8
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable*
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_1*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable_2*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_3*
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
T0*
use_nesterov( *
_class
loc:@Variable_4*
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
use_nesterov( *
_class
loc:@Variable_5*
use_locking( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
_output_shapes
:	�*
T0*
use_nesterov( *
_class
loc:@Variable_6*
use_locking( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
_output_shapes
:*
T0*
use_nesterov( *
_class
loc:@Variable_7*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
v
ArgMaxArgMaxadd_2ArgMax/dimension*#
_output_shapes
:���������*
T0*
output_type0	*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*#
_output_shapes
:���������*
T0*
output_type0	*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:���������*

SrcT0

Q
Const_6Const*
_output_shapes
:*
dtype0*
valueB: 
_
accuracyMeanCast_1Const_6*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
Z
accuracy_1/tagsConst*
_output_shapes
: *
dtype0*
valueB B
accuracy_1
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
: ""
train_op

Adam"�
trainable_variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
?
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_1:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_2:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_3:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_4:0"�
	variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
?
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_1:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_2:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_3:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_4:0
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0".
	summaries!

cross_entropy:0
accuracy_1:0~`�4       ^3\	�{�z���A*)

cross_entropy�/;A


accuracy_1�Q�>z�6       OW��	!:{���A2*)

cross_entropy1�)A


accuracy_1)\?l�}*6       OW��	�mr{���Ad*)

cross_entropyS�A


accuracy_1333?��)37       ���Y	=��{���A�*)

cross_entropy��A


accuracy_1R�?��7       ���Y	���{���A�*)

cross_entropy��A


accuracy_1333?��<H7       ���Y	+�|���A�*)

cross_entropy~+ A


accuracy_1333?�Q�R7       ���Y	\O|���A�*)

cross_entropyD#�@


accuracy_1R�?�x�7       ���Y	G��|���A�*)

cross_entropy�@�@


accuracy_1{.?��.W7       ���Y	>�|���A�*)

cross_entropy9��@


accuracy_1�p=?�`<�7       ���Y	���|���A�*)

cross_entropy��@


accuracy_1�Q8?�~l�7       ���Y	�)}���A�*)

cross_entropy8!�@


accuracy_1��(?%��K7       ���Y	�t}���A�*)

cross_entropy�V�@


accuracy_1�G?k
� 7       ���Y	��}���A�*)

cross_entropy�o�@


accuracy_1�Q8?j�� 7       ���Y	H��}���A�*)

cross_entropy\O�@


accuracy_1��??�1M7       ���Y	�O~���A�*)

cross_entropy)��@


accuracy_1\�B?��JE7       ���Y	�N~���A�*)

cross_entropy�G�@


accuracy_1{.?��E�7       ���Y	��~���A�*)

cross_entropy���@


accuracy_1�p=?H�27       ���Y	M�~���A�*)

cross_entropy�^�@


accuracy_1��(?8���7       ���Y	�C�~���A�*)

cross_entropyҊ�@


accuracy_1��(?�9s�7       ���Y	�R(���A�*)

cross_entropy�|v@


accuracy_1��Q?M�b�7       ���Y	��^���A�*)

cross_entropy&�q@


accuracy_1333?i^07       ���Y	٪���A�*)

cross_entropy�oh@


accuracy_1333?$xu-7       ���Y	7����A�*)

cross_entropy�>^@


accuracy_1��L?�Q��7       ���Y	-�����A�*)

cross_entropyr(W@


accuracy_1
�#?�C�7       ���Y	7:N����A�	*)

cross_entropy��F@


accuracy_1�Q8?���V7       ���Y	�o�����A�	*)

cross_entropy9�A@


accuracy_1��(?���7       ���Y	QI�����A�
*)

cross_entropy}9@


accuracy_1�p=?�\��7       ���Y	RY󀦑�A�
*)

cross_entropy�,4@


accuracy_1{.?,�$97       ���Y	s�)����A�
*)

cross_entropy|t@


accuracy_1=
W?�7�7       ���Y	{`����A�*)

cross_entropy�@


accuracy_1\�B?��=�7       ���Y	%������A�*)

cross_entropy�R@


accuracy_1�G?��=L7       ���Y	�ځ���A�*)

cross_entropy�B@


accuracy_1\�B?B��7       ���Y	�o����A�*)

cross_entropy;@


accuracy_1\�B?}��M7       ���Y	�G����A�*)

cross_entropy��@


accuracy_1�p=?�V��7       ���Y	�����A�*)

cross_entropy�y�?


accuracy_1\�B?aY17       ���Y	�e�����A�*)

cross_entropyf�?


accuracy_1\�B?���7       ���Y	�l낦��A�*)

cross_entropy���?


accuracy_1N�K?���7       ���Y	,"����A�*)

cross_entropyg��?


accuracy_1��Q?����7       ���Y	s&X����A�*)

cross_entropy�?�?


accuracy_1\�B?߿�7       ���Y	N�����A�*)

cross_entropy�k�?


accuracy_1�p=?&�}�7       ���Y	�̃���A�*)

cross_entropy
��?


accuracy_1\�B?�$'.7       ���Y	�q����A�*)

cross_entropy^��?


accuracy_1333?�77       ���Y	ȓJ����A�*)

cross_entropy�H�?


accuracy_1333?}�]�7       ���Y	�G�����A�*)

cross_entropy�E�?


accuracy_1�p=?��7       ���Y	�G�����A�*)

cross_entropy�&�?


accuracy_1333?m���7       ���Y	��ꄦ��A�*)

cross_entropy��?


accuracy_1=
W?�!�^7       ���Y	\� ����A�*)

cross_entropy�4�?


accuracy_1��Q?_�A,7       ���Y	��V����A�*)

cross_entropy�L�?


accuracy_1333?��n7       ���Y	�ً����A�*)

cross_entropy�k�?


accuracy_1�Q8?� 7       ���Y	������A�*)

cross_entropy��?


accuracy_1333?�sC�7       ���Y	_�����A�*)

cross_entropy���?


accuracy_1�G?���L7       ���Y	o?����A�*)

cross_entropy"̍?


accuracy_1�Q8?�'��7       ���Y	;ou����A�*)

cross_entropyz��?


accuracy_1=
W?@�@�7       ���Y	/x�����A�*)

cross_entropy�ȇ?


accuracy_1�G?(9��7       ���Y	w�߆���A�*)

cross_entropyP�?


accuracy_1�G?��#7       ���Y	\�����A�*)

cross_entropy�P�?


accuracy_1�G?�/87       ���Y	sK����A�*)

cross_entropy+�?


accuracy_1n�6?�'C7       ���Y	�e�����A�*)

cross_entropy��i?


accuracy_1��Q?��F|7       ���Y	�|�����A�*)

cross_entropy`e?


accuracy_1��Q?Ȍ	7       ���Y	"g쇦��A�*)

cross_entropy��h?


accuracy_1\�B?�Ր�7       ���Y	�k!����A�*)

cross_entropy�g?


accuracy_1�p=?�18e7       ���Y	��k����A�*)

cross_entropy:�?


accuracy_1)\?ꚫ 7       ���Y	����A�*)

cross_entropy�bM?


accuracy_1=
W?�U]�7       ���Y	�و���A�*)

cross_entropyG�X?


accuracy_1\�B?���7       ���Y	4����A�*)

cross_entropy���?


accuracy_1�z?T|��7       ���Y	�7F����A�*)

cross_entropy�?O?


accuracy_1�(\?hnv�7       ���Y	��|����A�*)

cross_entropy�G?


accuracy_1��Q?��>�7       ���Y	�J�����A�*)

cross_entropyNtJ?


accuracy_1\�B?�;�7       ���Y	��񉦑�A�*)

cross_entropyn/L?


accuracy_1\�B? ��7       ���Y	�+����A�*)

cross_entropy6�??


accuracy_1��Q?k^�7       ���Y	�b����A�*)

cross_entropy�<?


accuracy_1333?�3a�7       ���Y	Wŭ����A�*)

cross_entropyt2?


accuracy_1��Q?8Ã�7       ���Y	w�劦��A�*)

cross_entropy�7D?


accuracy_1�G?{�r7       ���Y	�n����A�*)

cross_entropy�5?


accuracy_1��Q?���7       ���Y	�T����A�*)

cross_entropy��8?


accuracy_1��L?J�?]7       ���Y	������A�*)

cross_entropy0I3?


accuracy_1�p=?�`�7       ���Y	ĈË���A�*)

cross_entropyAu0?


accuracy_1�G?��z>7       ���Y	�4�����A�*)

cross_entropy�+?


accuracy_1�G?�c_L7       ���Y	�"3����A�*)

cross_entropy@x6?


accuracy_1=
W?:7       ���Y	P�k����A�*)

cross_entropys�8?


accuracy_1�p=?f�`�7       ���Y	�������A�*)

cross_entropyp�.?


accuracy_1��L?�o�n7       ���Y	�쌦��A�*)

cross_entropy��:?


accuracy_1{.?����7       ���Y	��#����A� *)

cross_entropy<�0?


accuracy_1�p=?nl �7       ���Y	�\����A� *)

cross_entropy#�(?


accuracy_1�Q8?'���7       ���Y	,�����A� *)

cross_entropyH[?


accuracy_1=
W?@�:o7       ���Y	]̍���A�!*)

cross_entropyW�)?


accuracy_1�p=?��b7       ���Y	������A�!*)

cross_entropy�)?


accuracy_1=
W?���7       ���Y	w�>����A�!*)

cross_entropy�D"?


accuracy_1�G?�M��7       ���Y	�ex����A�"*)

cross_entropy
}#?


accuracy_1^NA?�;�7       ���Y	I>�����A�"*)

cross_entropyF�$?


accuracy_1333?uj�x7       ���Y	r�莦��A�#*)

cross_entropy.�?


accuracy_1��L?�$��7       ���Y	)Q5����A�#*)

cross_entropy�a3?


accuracy_1{.?��7       ���Y	��m����A�#*)

cross_entropyZ?


accuracy_1��Q?CՎd7       ���Y	T������A�$*)

cross_entropy�?


accuracy_1�G?3�ڰ7       ���Y	�f㏦��A�$*)

cross_entropyY�?


accuracy_1�G?����7       ���Y	m����A�%*)

cross_entropy�C?


accuracy_1\�B?T�/j7       ���Y	r�V����A�%*)

cross_entropy�?


accuracy_1�Q8?�厫7       ���Y	U{�����A�%*)

cross_entropy�R?


accuracy_1=
W?-���7       ���Y	��ʐ���A�&*)

cross_entropyj?


accuracy_1��L?�>�7       ���Y	h�����A�&*)

cross_entropy��>


accuracy_1fff?��t7       ���Y	`�?����A�'*)

cross_entropy�(?


accuracy_1��L?[�?w7       ���Y	kǐ����A�'*)

cross_entropy7�?


accuracy_1�p=?"7,27       ���Y	�lБ���A�'*)

cross_entropy-?


accuracy_1�Q8?��W7       ���Y	������A�(*)

cross_entropyк?


accuracy_1�p=?��\�7       ���Y	scM����A�(*)

cross_entropy0�?


accuracy_1��Q?n��W7       ���Y	�ψ����A�)*)

cross_entropy��?


accuracy_1��L?�"�_7       ���Y	wBÒ���A�)*)

cross_entropy���>


accuracy_1�(\?RW(�7       ���Y	�������A�)*)

cross_entropy�)�>


accuracy_1=
W?CJx-7       ���Y	\�6����A�**)

cross_entropy��?


accuracy_1�p=?0*7       ���Y	^%q����A�**)

cross_entropy�?


accuracy_1�Q8?0�A�7       ���Y	y�����A�**)

cross_entropy��?


accuracy_1��Q?�]�+7       ���Y	�F�����A�+*)

cross_entropy�?


accuracy_1��Q?S��G7       ���Y	��2����A�+*)

cross_entropy%
?


accuracy_1��L?�0B�7       ���Y	�l����A�,*)

cross_entropyr� ?


accuracy_1�Q8?(�p7       ���Y	vW�����A�,*)

cross_entropy�� ?


accuracy_1��L?+�"�7       ���Y	��┦��A�,*)

cross_entropy�?


accuracy_1��L?��̀7       ���Y	������A�-*)

cross_entropyy	?


accuracy_1�p=?�@L7       ���Y	��W����A�-*)

cross_entropy��>


accuracy_1=
W?T�y�7       ���Y	�ۑ����A�.*)

cross_entropy$?


accuracy_1�G?�[;7       ���Y	T�̕���A�.*)

cross_entropy�w?


accuracy_1\�B?�;7       ���Y	�%����A�.*)

cross_entropyW��>


accuracy_1=
W?!��7       ���Y	��W����A�/*)

cross_entropy�^?


accuracy_1\�B?�'U�7       ���Y	������A�/*)

cross_entropy��?


accuracy_1�p=?��RT7       ���Y	��̖���A�0*)

cross_entropy� ?


accuracy_1��Q?���[7       ���Y	�!����A�0*)

cross_entropy�T?


accuracy_1\�B?gXs�7       ���Y	�?C����A�0*)

cross_entropy��?


accuracy_1��L?^���7       ���Y	������A�1*)

cross_entropy��>


accuracy_1��Q?��87       ���Y	�߼����A�1*)

cross_entropy��>


accuracy_1�(\?��g�7       ���Y	�m�����A�2*)

cross_entropy��?


accuracy_1\�B?��7       ���Y	/�2����A�2*)

cross_entropy&�?


accuracy_1�Q8?���7       ���Y	�Kl����A�2*)

cross_entropy:��>


accuracy_1�k?�W��7       ���Y	N������A�3*)

cross_entropyt�?


accuracy_1�G?�{7       ���Y	�c񘦑�A�3*)

cross_entropyH�?


accuracy_1333?�*�7       ���Y	> .����A�3*)

cross_entropyR?


accuracy_1333?��7       ���Y	��k����A�4*)

cross_entropy:��>


accuracy_1�(\?��Z_7       ���Y	\蓮���A�4*)

cross_entropy�??


accuracy_1�G?|�?s7       ���Y	Z晦��A�5*)

cross_entropy��>


accuracy_1=
W?\�07       ���Y	&�(����A�5*)

cross_entropyfQ�>


accuracy_1�Ga?�%ͺ7       ���Y	­h����A�5*)

cross_entropy�T�>


accuracy_1��Q?gAp�7       ���Y	�����A�6*)

cross_entropyZ�>


accuracy_1��Q?/�{7       ���Y	�蚦��A�6*)

cross_entropy�?


accuracy_1
�#?�!�7       ���Y	��<����A�7*)

cross_entropyPC�>


accuracy_1=
W?�0�7       ���Y	�sy����A�7*)

cross_entropyM!?


accuracy_1\�B?�N�7       ���Y	�׹����A�7*)

cross_entropy8��>


accuracy_1��Q?D�7       ���Y	O������A�8*)

cross_entropyx��>


accuracy_1��L?��a�7       ���Y	�0����A�8*)

cross_entropy.�>


accuracy_1�(\?B�K7       ���Y	�$p����A�9*)

cross_entropy��	?


accuracy_1�Q8?x�!�7       ���Y	?������A�9*)

cross_entropy4��>


accuracy_1\�B? w|7       ���Y	�e蜦��A�9*)

cross_entropyg��>


accuracy_1fff?U�¨7       ���Y	��#����A�:*)

cross_entropy�?


accuracy_1�p=?�3L�7       ���Y	�T_����A�:*)

cross_entropye��>


accuracy_1��L?x��,7       ���Y	֢�����A�:*)

cross_entropy�d?


accuracy_1�p=?-4�v7       ���Y	b�睦��A�;*)

cross_entropyGc?


accuracy_1\�B?2��7       ���Y	�!����A�;*)

cross_entropy��?


accuracy_1�G?W���7       ���Y	\����A�<*)

cross_entropy�?


accuracy_1�Q8?�u��7       ���Y	������A�<*)

cross_entropyr��>


accuracy_1��Q?�3(7       ���Y	[#̞���A�<*)

cross_entropyD�?


accuracy_1{.?G
XC7       ���Y	�1����A�=*)

cross_entropy
?


accuracy_1��L?�#��7       ���Y	uE@����A�=*)

cross_entropy�-?


accuracy_1\�B?�:��7       ���Y	2�|����A�>*)

cross_entropy�:?


accuracy_1��Q?�4