       �K"	  @e2��Abrain.Event:2��0���      �*�	�'`e2��A"ݗ
l
xPlaceholder* 
shape:���������d*
dtype0*+
_output_shapes
:���������d
e
y_Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
f
Reshape/shapeConst*%
valueB"����   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*/
_output_shapes
:���������d*
Tshape0
o
truncated_normal/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
R
ConstConst*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_1
VariableV2*
shape:d*
	container *
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
|
BiasAddBiasAddConv2DVariable_1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
q
truncated_normal_1/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
T
Const_1Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_3
VariableV2*
shape:d*
	container *
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
q
truncated_normal_2/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
w
Variable_4/readIdentity
Variable_4*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
T
Const_2Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_5
VariableV2*
shape:d*
	container *
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
T0*

Tidx0*
N*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
valueB"����,  *
dtype0*
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*(
_output_shapes
:����������*
Tshape0
N
	keep_probPlaceholder*
shape:*
dtype0*
_output_shapes
:
V
dropout/ShapeShape	Reshape_1*
T0*
_output_shapes
:*
out_type0
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
T0*
seed2 *

seed *
dtype0*(
_output_shapes
:����������
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
valueB",     *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
seed2 *

seed *
dtype0*
_output_shapes
:	�
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
VariableV2*
shape:	�*
	container *
_output_shapes
:	�*
dtype0*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
T
Const_3Const*
valueB*���=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
shape:*
	container *
_output_shapes
:*
dtype0*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes
:*
_class
loc:@Variable_7
�
MatMulMatMuldropout/mulVariable_6/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
U
addAddMatMulVariable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
H
ShapeShapeadd*
T0*
_output_shapes
:*
out_type0
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
J
Shape_1Shapeadd*
T0*
_output_shapes
:*
out_type0
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
T0*
N*
_output_shapes
:*

axis 
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
T0*

Tidx0*
N*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
T0*0
_output_shapes
:������������������*
Tshape0
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
T0*
_output_shapes
:*
out_type0
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
T0*
N*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_2/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
T0*

Tidx0*
N*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
T0*0
_output_shapes
:������������������*
Tshape0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*
T0*
N*
_output_shapes
:*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*#
_output_shapes
:���������*
Tshape0
Q
Const_4Const*
valueB: *
dtype0*
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
valueB Bcross_entropy*
dtype0*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
T0*
_output_shapes
:*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
T0*
_output_shapes
:*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:*
out_type0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
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
a
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
_output_shapes
:*
out_type0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
_output_shapes
:*
out_type0
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
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
T0*'
_output_shapes
:���������*
Tshape0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	�*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*#
_output_shapes
:���������*
out_type0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*#
_output_shapes
:���������*
out_type0
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
T0*
_output_shapes
:*
Tshape0
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
T0*
_output_shapes
:*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
T0*
_output_shapes
:*
out_type0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*#
_output_shapes
:���������*
out_type0
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
T0*(
_output_shapes
:����������*
Tshape0
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
T0*
_output_shapes
:*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
T0*
_output_shapes
:*
out_type0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*0
_output_shapes
:����������*
Tshape0
\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
b
gradients/concat_grad/ShapeShapeMaxPool*
T0*
_output_shapes
:*
out_type0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
T0*
N*&
_output_shapes
:::*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*.
_class$
" loc:@gradients/concat_grad/Slice
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_1
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_2
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:���������d
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
T0*
N* 
_output_shapes
::*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
T0*
N* 
_output_shapes
::*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
T0*
N* 
_output_shapes
::*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@Variable
{
beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: *
_class
loc:@Variable
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable
�
Variable/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable_1/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
!Variable_3/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
!Variable_4/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Variable_5/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
!Variable_6/Adam/Initializer/zerosConst*
valueB	�*    *
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
#Variable_6/Adam_1/Initializer/zerosConst*
valueB	�*    *
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7
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
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7
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
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_7
W
Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *&
_output_shapes
:dd*
_class
loc:@Variable
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_1
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *&
_output_shapes
:dd*
_class
loc:@Variable_2
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_3
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *&
_output_shapes
:dd*
_class
loc:@Variable_4
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_5
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:	�*
_class
loc:@Variable_6
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:*
_class
loc:@Variable_7
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
Q
Const_5Const*
valueB: *
dtype0*
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
accuracy_1/tagsConst*
valueB B
accuracy_1*
dtype0*
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
: "�L��      eef�	�mve2��AJ��
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
shape:���������d*
dtype0*+
_output_shapes
:���������d
e
y_Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
f
Reshape/shapeConst*%
valueB"����   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *
shared_name *
dtype0*&
_output_shapes
:dd
�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
R
ConstConst*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_1
VariableV2*
shape:d*
	container *
shared_name *
dtype0*
_output_shapes
:d
�
Variable_1/AssignAssign
Variable_1Const*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
|
BiasAddBiasAddConv2DVariable_1/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
q
truncated_normal_1/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *
shared_name *
dtype0*&
_output_shapes
:dd
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
T
Const_1Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_3
VariableV2*
shape:d*
	container *
shared_name *
dtype0*
_output_shapes
:d
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
q
truncated_normal_2/shapeConst*%
valueB"   d      d   *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
seed2 *

seed *
dtype0*&
_output_shapes
:dd
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
VariableV2*
shape:dd*
	container *
shared_name *
dtype0*&
_output_shapes
:dd
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
w
Variable_4/readIdentity
Variable_4*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
T
Const_2Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_5
VariableV2*
shape:d*
	container *
shared_name *
dtype0*
_output_shapes
:d
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*
data_formatNHWC*/
_output_shapes
:���������d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
T0*

Tidx0*
N*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
valueB"����,  *
dtype0*
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
shape:*
dtype0*
_output_shapes
:
V
dropout/ShapeShape	Reshape_1*
T0*
_output_shapes
:*
out_type0
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
T0*
seed2 *

seed *
dtype0*(
_output_shapes
:����������
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
valueB",     *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
seed2 *

seed *
dtype0*
_output_shapes
:	�
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
VariableV2*
shape:	�*
	container *
shared_name *
dtype0*
_output_shapes
:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
T
Const_3Const*
valueB*���=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
shape:*
	container *
shared_name *
dtype0*
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes
:*
_class
loc:@Variable_7
�
MatMulMatMuldropout/mulVariable_6/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
U
addAddMatMulVariable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
H
ShapeShapeadd*
T0*
_output_shapes
:*
out_type0
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
J
Shape_1Shapeadd*
T0*
_output_shapes
:*
out_type0
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
T0*
N*
_output_shapes
:*

axis 
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
T0*

Tidx0*
N*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
T0*
Tshape0*0
_output_shapes
:������������������
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
T0*
_output_shapes
:*
out_type0
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
T0*
N*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_2/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
T0*

Tidx0*
N*
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
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*
T0*
N*
_output_shapes
:*

axis 
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
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
^
MeanMean	Reshape_4Const_4*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
`
cross_entropy/tagsConst*
valueB Bcross_entropy*
dtype0*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
T0*
_output_shapes
:*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
T0*
_output_shapes
:*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:*
out_type0
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
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
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
a
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
_output_shapes
:*
out_type0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
^
gradients/add_grad/ShapeShapeMatMul*
T0*
_output_shapes
:*
out_type0
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
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

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
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
T0*'
_output_shapes
:���������*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*(
_output_shapes
:����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	�*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*#
_output_shapes
:���������*
out_type0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*#
_output_shapes
:���������*
out_type0
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

Tidx0*
	keep_dims( *
_output_shapes
:
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

Tidx0*
	keep_dims( *
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
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
T0*
_output_shapes
:*
out_type0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*#
_output_shapes
:���������*
out_type0
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

Tidx0*
	keep_dims( *
_output_shapes
:
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

Tidx0*
	keep_dims( *
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
T0*(
_output_shapes
:����������*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
T0*
_output_shapes
:*
out_type0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:����������
\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
b
gradients/concat_grad/ShapeShapeMaxPool*
T0*
_output_shapes
:*
out_type0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
T0*
N*&
_output_shapes
:::*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
T0*
Index0*J
_output_shapes8
6:4������������������������������������
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*.
_class$
" loc:@gradients/concat_grad/Slice
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_1
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_2
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
T0*
paddingVALID*
strides
*
data_formatNHWC*/
_output_shapes
:���������d
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*/
_output_shapes
:���������d
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
T0*
N* 
_output_shapes
::*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
T0*
N* 
_output_shapes
::*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
T0*
N* 
_output_shapes
::*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingVALID*
strides
*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@Variable
{
beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: *
_class
loc:@Variable
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable
�
Variable/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable_1/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_1*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_2*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_2
�
!Variable_3/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_3*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_3
�
!Variable_4/Adam/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*%
valueBdd*    *
dtype0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_4*
shape:dd*
shared_name *&
_output_shapes
:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*
_class
loc:@Variable_5
�
Variable_5/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_5*
shape:d*
shared_name *
_output_shapes
:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_5
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
!Variable_6/Adam/Initializer/zerosConst*
valueB	�*    *
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
#Variable_6/Adam_1/Initializer/zerosConst*
valueB	�*    *
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@Variable_6*
shape:	�*
shared_name *
_output_shapes
:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_6
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_output_shapes
:	�*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7
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
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7
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
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_7
W
Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable*
use_locking( *&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_1*
use_locking( *
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_2*
use_locking( *&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
use_locking( *
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_4*
use_locking( *&
_output_shapes
:dd*
use_nesterov( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_5*
use_locking( *
_output_shapes
:d*
use_nesterov( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_6*
use_locking( *
_output_shapes
:	�*
use_nesterov( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
use_locking( *
_output_shapes
:*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
_
accuracyMeanCast_1Const_5*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
dtype0*
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
: ""�
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0"�
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
accuracy_1:0"
train_op

Adam��f�4       ^3\	A�e2��A*)

cross_entropy�c�?


accuracy_1�?��S6       OW��	Є
f2��A2*)

cross_entropy�?


accuracy_1q=
?y�X6       OW��	Sf2��Ad*)

cross_entropy�yX?


accuracy_1333?��	�7       ���Y	Kߝf2��A�*)

cross_entropysR�?


accuracy_1R�?�ÿ�7       ���Y	݉�f2��A�*)

cross_entropy��+?


accuracy_1333?�q:�7       ���Y	Be6g2��A�*)

cross_entropy1�(?


accuracy_1�G?:#!7       ���Y	���g2��A�*)

cross_entropy���>


accuracy_1��Q?
t�b7       ���Y	��g2��A�*)

cross_entropy�;?


accuracy_1333?vf�47       ���Y	�Gh2��A�*)

cross_entropy]+?


accuracy_1��(?��Ye7       ���Y	��ah2��A�*)

cross_entropy��G?


accuracy_1�Q8?�&�7       ���Y	x�h2��A�*)

cross_entropy}4?


accuracy_1\�B?�A۸7       ���Y	qi2��A�*)

cross_entropy��\?


accuracy_1333?E^�7       ���Y	bi2��A�*)

cross_entropy`�+?


accuracy_1�G?%`��7       ���Y	Gۮi2��A�*)

cross_entropy�(;?


accuracy_1333?J�;'7       ���Y	���i2��A�*)

cross_entropy�	?


accuracy_1�G?��Cj7       ���Y	�%Fj2��A�*)

cross_entropy,Wd?


accuracy_1{.?��y�7       ���Y	��j2��A�*)

cross_entropyʎ?


accuracy_1�p=?�b�7       ���Y	N�j2��A�*)

cross_entropy�j%?


accuracy_1�Q8?R��7       ���Y	��*k2��A�*)

cross_entropy�^?


accuracy_1333? :�J7       ���Y	��xk2��A�*)

cross_entropyE�?


accuracy_1�G?�d�7       ���Y	���k2��A�*)

cross_entropy���>


accuracy_1��L?�{�7       ���Y	�m-l2��A�*)

cross_entropy�?


accuracy_1�p=?��c7       ���Y	�xl2��A�*)

cross_entropy�3;?


accuracy_1333?�Zϑ7       ���Y	�=�l2��A�*)

cross_entropyє?


accuracy_1�G?k}��7       ���Y	�m2��A�	*)

cross_entropy0�?


accuracy_1�G?���7       ���Y	T�cm2��A�	*)

cross_entropy/�?


accuracy_1{.?t��7       ���Y	���m2��A�
*)

cross_entropy���>


accuracy_1\�B?(��)7       ���Y	a8n2��A�
*)

cross_entropy��?


accuracy_1�Q8?�z��7       ���Y	�bTn2��A�
*)

cross_entropyʋ�>


accuracy_1�G?ȅ97       ���Y	\��n2��A�*)

cross_entropy�
�>


accuracy_1��L?��7       ���Y	� �n2��A�*)

cross_entropy#X�>


accuracy_1�G?ǉ;	7       ���Y	; co2��A�*)

cross_entropy�[�>


accuracy_1\�B?s~ԝ7       ���Y	�W�o2��A�*)

cross_entropy��>


accuracy_1��Q?1�F�7       ���Y	~� p2��A�*)

cross_entropy�[�>


accuracy_1��L?^�s%7       ���Y	�JVp2��A�*)

cross_entropyd��>


accuracy_1\�B?)=<7       ���Y	���p2��A�*)

cross_entropy�?


accuracy_1��Q?N� ,7       ���Y	fw�p2��A�*)

cross_entropy�?


accuracy_1^NA? �7       ���Y	w�Sq2��A�*)

cross_entropy�t?


accuracy_1�G?PH��7       ���Y	-��q2��A�*)

cross_entropy%@�>


accuracy_1��Q?`�}g7       ���Y	���q2��A�*)

cross_entropyv��>


accuracy_1�p=?Y�.7       ���Y	�aSr2��A�*)

cross_entropyt�>


accuracy_1�G?���7       ���Y	Ga�r2��A�*)

cross_entropy�7?


accuracy_1�G?<e�7       ���Y	�Us2��A�*)

cross_entropy���>


accuracy_1fff?ǍQ>7       ���Y	�Dhs2��A�*)

cross_entropy�'�>


accuracy_1��L?	T�C7       ���Y	��s2��A�*)

cross_entropy�h�>


accuracy_1�G?�>K�7       ���Y	�	t2��A�*)

cross_entropyV�>


accuracy_1�Ga?:�7       ���Y	Y�\t2��A�*)

cross_entropy�L�>


accuracy_1�(\?QU��7       ���Y	�űt2��A�*)

cross_entropy���>


accuracy_1�G?H枊7       ���Y	�u2��A�*)

cross_entropy��>


accuracy_1�G?�[~r7       ���Y	~�Xu2��A�*)

cross_entropyV��>


accuracy_1��Q?@���7       ���Y	�`�u2��A�*)

cross_entropy�п>


accuracy_1=
W?���7       ���Y	4�v2��A�*)

cross_entropy0f�>


accuracy_1�(\?Ͳ7       ���Y	Bdv2��A�*)

cross_entropy�>


accuracy_1=
W?�V��7       ���Y	�\�v2��A�*)

cross_entropy���>


accuracy_1\�B?[� 7       ���Y	@�w2��A�*)

cross_entropyd��>


accuracy_1�(\?`W�z7       ���Y	k�dw2��A�*)

cross_entropy:��>


accuracy_1�Q8?�!t�7       ���Y	i�w2��A�*)

cross_entropy���>


accuracy_1�<?�&g7       ���Y	��	x2��A�*)

cross_entropy���>


accuracy_1=
W?��237       ���Y	��[x2��A�*)

cross_entropy�2�>


accuracy_1�G?�l�7       ���Y	���x2��A�*)

cross_entropy���>


accuracy_1��L?�Q7       ���Y	�\y2��A�*)

cross_entropy�ƪ>


accuracy_1�Ga?��.7       ���Y	&�zy2��A�*)

cross_entropy�?


accuracy_1�p=?���37       ���Y	���y2��A�*)

cross_entropy(5�>


accuracy_1�Ga?{bzV7       ���Y	�&z2��A�*)

cross_entropy�V�>


accuracy_1�(\??V:b7       ���Y	>{z2��A�*)

cross_entropyad?


accuracy_1�G?��>�7       ���Y	ί�z2��A�*)

cross_entropyX��>


accuracy_1��L?#k�77       ���Y	��%{2��A�*)

cross_entropy���>


accuracy_1=
W?wX8�7       ���Y	uLu{2��A�*)

cross_entropy���>


accuracy_1�k?����7       ���Y	���{2��A�*)

cross_entropy�L�>


accuracy_1��Q?n,�7       ���Y	�O|2��A�*)

cross_entropyȫ�>


accuracy_1�(\?��57       ���Y	��m|2��A�*)

cross_entropy��>


accuracy_1=
W?��7       ���Y	"u�|2��A�*)

cross_entropy|f�>


accuracy_1�(\?�z2a7       ���Y	t)}2��A�*)

cross_entropy���>


accuracy_1=
W?Iq�#7       ���Y	�|}2��A�*)

cross_entropyb�>


accuracy_1�k?�9�|7       ���Y	<��}2��A�*)

cross_entropy4�>


accuracy_1fff?i���7       ���Y	�~2��A�*)

cross_entropy�m�>


accuracy_1��Q?�*$�7       ���Y	��s~2��A�*)

cross_entropyc�>


accuracy_1�Ga?j
R�7       ���Y	z5�~2��A�*)

cross_entropy��>


accuracy_1�Ga?���7       ���Y	92��A�*)

cross_entropy��>


accuracy_1�(\?����7       ���Y	�2j2��A�*)

cross_entropy3��>


accuracy_1�(\?2��7       ���Y	�ú2��A�*)

cross_entropy|��>


accuracy_1�Ga?1��7       ���Y	�#&�2��A�*)

cross_entropy\r�>


accuracy_1=
W?����7       ���Y	�7x�2��A� *)

cross_entropy��>


accuracy_1fff?�H;=7       ���Y	�ɀ2��A� *)

cross_entropy�a�>


accuracy_1��Q?���&7       ���Y	u��2��A� *)

cross_entropy�Ii>


accuracy_1fff?;�E7       ���Y	��l�2��A�!*)

cross_entropy��>


accuracy_1��Q?��7       ���Y	�R��2��A�!*)

cross_entropyl�&>


accuracy_1��u?�/yi7       ���Y	,C�2��A�!*)

cross_entropy�,�>


accuracy_1�k?8���7       ���Y	�q^�2��A�"*)

cross_entropyђc>


accuracy_1k?�@JN7       ���Y	�7��2��A�"*)

cross_entropy쬗>


accuracy_1��Q?�(�7       ���Y	+���2��A�#*)

cross_entropy���>


accuracy_1�Ga?�> 7       ���Y	�9[�2��A�#*)

cross_entropy��>


accuracy_1�k?7��7       ���Y	F��2��A�#*)

cross_entropy%��>


accuracy_1�k?ܞ|�7       ���Y	����2��A�$*)

cross_entropy*p>


accuracy_1�k?b�&7       ���Y	Y�E�2��A�$*)

cross_entropyE{|>


accuracy_1�Ga?���7       ���Y	"��2��A�%*)

cross_entropy@�>


accuracy_1�Ga?��Ix7       ���Y	�0�2��A�%*)

cross_entropy-�>


accuracy_1ףp?���7       ���Y	��/�2��A�%*)

cross_entropy�M>


accuracy_1��u?gw-7       ���Y	��}�2��A�&*)

cross_entropy� j>


accuracy_1ףp?�>z7       ���Y	��˅2��A�&*)

cross_entropyJ�$>


accuracy_1��u?Ҭ��7       ���Y	ã�2��A�'*)

cross_entropy��>


accuracy_1=
W?L�*�7       ���Y	����2��A�'*)

cross_entropy7t>


accuracy_1ףp?0N�/7       ���Y	�	ކ2��A�'*)

cross_entropy�O�>


accuracy_1�(\?>^h7       ���Y	ʘ0�2��A�(*)

cross_entropy\�S>


accuracy_1ףp?ZI%7       ���Y	{@��2��A�(*)

cross_entropy�H@>


accuracy_1ףp?u�#�7       ���Y	��ڇ2��A�)*)

cross_entropy�
l>


accuracy_1fff?=���7       ���Y	7�*�2��A�)*)

cross_entropy�%>


accuracy_1��u?�Ղ7       ���Y	�Q~�2��A�)*)

cross_entropy�\v>


accuracy_1ףp?���c7       ���Y	4�҈2��A�**)

cross_entropy+W�>


accuracy_1fff?�6n�7       ���Y	P)�2��A�**)

cross_entropy�+{>


accuracy_1fff?�A��7       ���Y	�1|�2��A�**)

cross_entropy�kg>


accuracy_1�k?���7       ���Y	�0�2��A�+*)

cross_entropyf8V>


accuracy_1ףp?���7       ���Y	!g?�2��A�+*)

cross_entropy�Cr>


accuracy_1�k?�o�7       ���Y	B��2��A�,*)

cross_entropy��>


accuracy_1�Ga?+�7       ���Y	���2��A�,*)

cross_entropya�l>


accuracy_1ףp?_�RQ7       ���Y	T�B�2��A�,*)

cross_entropy�_�>


accuracy_1fff?���7       ���Y	�u��2��A�-*)

cross_entropy}�F>


accuracy_1ףp?+
&�7       ���Y	�2��A�-*)

cross_entropy�S2>


accuracy_1��u?���7       ���Y	��K�2��A�.*)

cross_entropy]��>


accuracy_1fff?~�,7       ���Y	FF��2��A�.*)

cross_entropy��v>


accuracy_1fff??�k7       ���Y	���2��A�.*)

cross_entropy�^>


accuracy_1�k?,/7       ���Y	��e�2��A�/*)

cross_entropy�(�>


accuracy_1�Ga?L7       ���Y	fF��2��A�/*)

cross_entropy��_>


accuracy_1ףp?��(7       ���Y	��2��A�0*)

cross_entropy�DE>


accuracy_1��u?~,�7       ���Y	U c�2��A�0*)

cross_entropyv\=>


accuracy_1  �?2�3`7       ���Y	(ȷ�2��A�0*)

cross_entropy��>


accuracy_1�k?�hV7       ���Y	���2��A�1*)

cross_entropy�_U>


accuracy_1ףp?�V�p7       ���Y	^_�2��A�1*)

cross_entropy�][>


accuracy_1��u?��7       ���Y	�Բ�2��A�2*)

cross_entropy�U]>


accuracy_1ףp?;ɦY7       ���Y	���2��A�2*)

cross_entropya��>


accuracy_1�Ga?#���7       ���Y	��\�2��A�2*)

cross_entropyM/->


accuracy_1H�z?�G�7       ���Y	Ɛ2��A�3*)

cross_entropy|<2>


accuracy_1H�z?�0��7       ���Y	���2��A�3*)

cross_entropy�}�>


accuracy_1fff?/:��7       ���Y	 �k�2��A�3*)

cross_entropyXގ>


accuracy_1fff?u4i7       ���Y	;A��2��A�4*)

cross_entropy��*>


accuracy_1H�z?~ۃ�7       ���Y	�]�2��A�4*)

cross_entropy�7p>


accuracy_1�k?��@e7       ���Y	n(c�2��A�5*)

cross_entropy2d'>


accuracy_1  �?���7       ���Y	�䵒2��A�5*)

cross_entropy4�>


accuracy_1H�z?J�7       ���Y	-1�2��A�5*)

cross_entropy�>


accuracy_1H�z?��J7       ���Y	4�Z�2��A�6*)

cross_entropy�r%>


accuracy_1ףp?�GN7       ���Y	oE��2��A�6*)

cross_entropyu߇>


accuracy_1�k?V���7       ���Y	x��2��A�7*)

cross_entropy})>


accuracy_1H�z?:�Ձ7       ���Y	aKp�2��A�7*)

cross_entropy�oP>


accuracy_1ףp?��7       ���Y	�O��2��A�7*)

cross_entropy�ZF>


accuracy_1�k?7��X7       ���Y	 ��2��A�8*)

cross_entropy��>


accuracy_1  �?t}�7       ���Y	��g�2��A�8*)

cross_entropy�0$>


accuracy_1��u?�K�7       ���Y	6��2��A�9*)

cross_entropy�IA>


accuracy_1��u?��:]7       ���Y	VI�2��A�9*)

cross_entropy��A>


accuracy_1��u?b��7       ���Y	�(_�2��A�9*)

cross_entropyP�>


accuracy_1H�z?n�9t7       ���Y	N��2��A�:*)

cross_entropy��I>


accuracy_1H�z?N�M[7       ���Y	lh�2��A�:*)

cross_entropyj�>


accuracy_1��u?}��Q7       ���Y	OKo�2��A�:*)

cross_entropy�:>


accuracy_1H�z?Sy�7       ���Y	���2��A�;*)

cross_entropyO?U>


accuracy_1��u?����7       ���Y	��2��A�;*)

cross_entropy��5>


accuracy_1  �?���7       ���Y	h�d�2��A�<*)

cross_entropy��>


accuracy_1��u?�pR�7       ���Y	���2��A�<*)

cross_entropy�->


accuracy_1H�z?o���7       ���Y	/�2��A�<*)

cross_entropyǷ>


accuracy_1��u?�<��7       ���Y	�W�2��A�=*)

cross_entropy�lA>


accuracy_1��u?f$��7       ���Y	(騙2��A�=*)

cross_entropy#P&>


accuracy_1ףp?��u7       ���Y	8���2��A�>*)

cross_entropy��8>


accuracy_1H�z?�8ҕ