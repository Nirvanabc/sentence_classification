       �K"	  @�֖�Abrain.Event:2���+     ��K�	%G�֖�A"��
l
xPlaceholder*
dtype0*+
_output_shapes
:���������d* 
shape:���������d
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*/
_output_shapes
:���������d*
Tshape0
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
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
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable*
validate_shape(
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
R
ConstConst*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_1
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_1/AssignAssign
Variable_1Const*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_1*
validate_shape(
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
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

|
BiasAddBiasAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      d   �   *
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:d�
�

Variable_2
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_2*
validate_shape(
x
Variable_2/readIdentity
Variable_2*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_2
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_3*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:�*
_class
loc:@Variable_3
�
Conv2D_1Conv2DReluVariable_2/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:����������
�
MaxPoolMaxPoolRelu_1*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   d      d   *
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
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
:dd
�

Variable_4
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_4*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
T
Const_2Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_5
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_5*
validate_shape(
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
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
q
truncated_normal_3/shapeConst*
dtype0*%
valueB"      d   �   *
_output_shapes
:
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
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*'
_output_shapes
:d�
�

Variable_6
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_6*
validate_shape(
x
Variable_6/readIdentity
Variable_6*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_6
V
Const_3Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_7
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_7*
validate_shape(
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:�*
_class
loc:@Variable_7
�
Conv2D_3Conv2DRelu_2Variable_6/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:����������
�
	MaxPool_1MaxPoolRelu_3*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

q
truncated_normal_4/shapeConst*
dtype0*%
valueB"   d      d   *
_output_shapes
:
\
truncated_normal_4/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_4/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*&
_output_shapes
:dd
�

Variable_8
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_8*
validate_shape(
w
Variable_8/readIdentity
Variable_8*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_8
T
Const_4Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_9
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_9/AssignAssign
Variable_9Const_4*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_9*
validate_shape(
k
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
�
Conv2D_4Conv2DReshapeVariable_8/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

�
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_4Relu	BiasAdd_4*
T0*/
_output_shapes
:���������d
q
truncated_normal_5/shapeConst*
dtype0*%
valueB"      d   �   *
_output_shapes
:
\
truncated_normal_5/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_5/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*'
_output_shapes
:d�
�
Variable_10
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_10/AssignAssignVariable_10truncated_normal_5*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_10*
validate_shape(
{
Variable_10/readIdentityVariable_10*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_10
V
Const_5Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
y
Variable_11
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_11/AssignAssignVariable_11Const_5*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_11*
validate_shape(
o
Variable_11/readIdentityVariable_11*
T0*
_output_shapes	
:�*
_class
loc:@Variable_11
�
Conv2D_5Conv2DRelu_4Variable_10/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_5BiasAddConv2D_5Variable_11/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_5Relu	BiasAdd_5*
T0*0
_output_shapes
:����������
�
	MaxPool_2MaxPoolRelu_5*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
T0*

Tidx0*
N*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
dtype0*
valueB"����X  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*(
_output_shapes
:����������*
Tshape0
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
T0*
out_type0*
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
T0*
dtype0*

seed *(
_output_shapes
:����������*
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
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
:����������
i
truncated_normal_6/shapeConst*
dtype0*
valueB"X     *
_output_shapes
:
\
truncated_normal_6/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_6/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
T0*
dtype0*

seed *
_output_shapes
:	�*
seed2 
�
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*
_output_shapes
:	�
�
Variable_12
VariableV2*
shared_name *
shape:	�*
_output_shapes
:	�*
	container *
dtype0
�
Variable_12/AssignAssignVariable_12truncated_normal_6*
T0*
_output_shapes
:	�*
use_locking(*
_class
loc:@Variable_12*
validate_shape(
s
Variable_12/readIdentityVariable_12*
T0*
_output_shapes
:	�*
_class
loc:@Variable_12
T
Const_6Const*
dtype0*
valueB*���=*
_output_shapes
:
w
Variable_13
VariableV2*
shared_name *
shape:*
_output_shapes
:*
	container *
dtype0
�
Variable_13/AssignAssignVariable_13Const_6*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@Variable_13*
validate_shape(
n
Variable_13/readIdentityVariable_13*
T0*
_output_shapes
:*
_class
loc:@Variable_13
�
MatMulMatMuldropout/mulVariable_12/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
V
addAddMatMulVariable_13/read*
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
ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
J
Shape_1Shapeadd*
T0*
out_type0*
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
T0*

axis *
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
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
T0*
out_type0*
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
T0*

axis *
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
T0*

axis *
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
T0*#
_output_shapes
:���������*
Tshape0
Q
Const_7Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_7*
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
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*.
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
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
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
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_12/read*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	�*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:���������
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
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
T0*
out_type0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
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
T0*(
_output_shapes
:����������*
Tshape0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:����������
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
:����������*5
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
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*0
_output_shapes
:����������*
Tshape0
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
gradients/concat_grad/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
T0*
out_type0*
N*&
_output_shapes
:::
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
T0*0
_output_shapes
:����������*.
_class$
" loc:@gradients/concat_grad/Slice
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_1
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_2
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradRelu_1MaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_5	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
gradients/Relu_1_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:����������
�
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:����������
�
gradients/Relu_5_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_5*
T0*0
_output_shapes
:����������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad
�
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradgradients/Relu_5_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_5_grad/tuple/group_depsNoOp^gradients/Relu_5_grad/ReluGrad%^gradients/BiasAdd_5_grad/BiasAddGrad
�
1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentitygradients/Relu_5_grad/ReluGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_5_grad/ReluGrad
�
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad
�
gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*'
_output_shapes
:d�*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
gradients/Conv2D_3_grad/ShapeNShapeNRelu_2Variable_6/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2 gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*'
_output_shapes
:d�*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
gradients/Conv2D_5_grad/ShapeNShapeNRelu_4Variable_10/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/read1gradients/BiasAdd_5_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4 gradients/Conv2D_5_grad/ShapeN:11gradients/BiasAdd_5_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput
�
2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*'
_output_shapes
:d�*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter
�
gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:���������d
�
gradients/Relu_4_grad/ReluGradReluGrad0gradients/Conv2D_5_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:���������d
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
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
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
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
�
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad
�
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

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
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

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
:dd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput
�
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?*
_class
loc:@Variable
�
beta1_power
VariableV2*
_output_shapes
: *
dtype0*
	container *
shared_name *
shape: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@Variable*
validate_shape(
g
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *w�?*
_class
loc:@Variable
�
beta2_power
VariableV2*
_output_shapes
: *
dtype0*
	container *
shared_name *
shape: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@Variable*
validate_shape(
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Variable/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable
�
Variable/Adam
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable*
validate_shape(
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable*
validate_shape(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable
�
!Variable_1/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_1*
validate_shape(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
#Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_1*
validate_shape(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_1
�
!Variable_2/Adam/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_2*
validate_shape(
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_2
�
#Variable_2/Adam_1/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_2*
validate_shape(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_2
�
!Variable_3/Adam/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_3*
validate_shape(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:�*
_class
loc:@Variable_3
�
#Variable_3/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_3*
validate_shape(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@Variable_3
�
!Variable_4/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_4*
validate_shape(
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
#Variable_4/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_4*
validate_shape(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_4
�
!Variable_5/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_5*
validate_shape(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
#Variable_5/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_5
�
Variable_5/Adam_1
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_5*
validate_shape(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_5
�
!Variable_6/Adam/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_6*
validate_shape(
�
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_6
�
#Variable_6/Adam_1/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_6*
validate_shape(
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_6
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_7*
validate_shape(
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:�*
_class
loc:@Variable_7
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_7*
validate_shape(
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@Variable_7
�
!Variable_8/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable_8
�
Variable_8/Adam
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable_8
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_8*
validate_shape(
�
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_8
�
#Variable_8/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*%
valueBdd*    *
_class
loc:@Variable_8
�
Variable_8/Adam_1
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd*
_class
loc:@Variable_8
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
use_locking(*
_class
loc:@Variable_8*
validate_shape(
�
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*&
_output_shapes
:dd*
_class
loc:@Variable_8
�
!Variable_9/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_9
�
Variable_9/Adam
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_9
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_9*
validate_shape(
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
�
#Variable_9/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
valueBd*    *
_class
loc:@Variable_9
�
Variable_9/Adam_1
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d*
_class
loc:@Variable_9
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
_class
loc:@Variable_9*
validate_shape(
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
�
"Variable_10/Adam/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_10
�
Variable_10/Adam
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_10
�
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_10*
validate_shape(
�
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_10
�
$Variable_10/Adam_1/Initializer/zerosConst*'
_output_shapes
:d�*
dtype0*&
valueBd�*    *
_class
loc:@Variable_10
�
Variable_10/Adam_1
VariableV2*'
_output_shapes
:d�*
dtype0*
	container *
shared_name *
shape:d�*
_class
loc:@Variable_10
�
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d�*
use_locking(*
_class
loc:@Variable_10*
validate_shape(
�
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*'
_output_shapes
:d�*
_class
loc:@Variable_10
�
"Variable_11/Adam/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_11
�
Variable_11/Adam
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_11
�
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_11*
validate_shape(
y
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_output_shapes	
:�*
_class
loc:@Variable_11
�
$Variable_11/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*
dtype0*
valueB�*    *
_class
loc:@Variable_11
�
Variable_11/Adam_1
VariableV2*
_output_shapes	
:�*
dtype0*
	container *
shared_name *
shape:�*
_class
loc:@Variable_11
�
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:�*
use_locking(*
_class
loc:@Variable_11*
validate_shape(
}
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_output_shapes	
:�*
_class
loc:@Variable_11
�
"Variable_12/Adam/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
valueB	�*    *
_class
loc:@Variable_12
�
Variable_12/Adam
VariableV2*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
shape:	�*
_class
loc:@Variable_12
�
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
T0*
_output_shapes
:	�*
use_locking(*
_class
loc:@Variable_12*
validate_shape(
}
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_output_shapes
:	�*
_class
loc:@Variable_12
�
$Variable_12/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
valueB	�*    *
_class
loc:@Variable_12
�
Variable_12/Adam_1
VariableV2*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
shape:	�*
_class
loc:@Variable_12
�
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	�*
use_locking(*
_class
loc:@Variable_12*
validate_shape(
�
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_output_shapes
:	�*
_class
loc:@Variable_12
�
"Variable_13/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@Variable_13
�
Variable_13/Adam
VariableV2*
_output_shapes
:*
dtype0*
	container *
shared_name *
shape:*
_class
loc:@Variable_13
�
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@Variable_13*
validate_shape(
x
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_13
�
$Variable_13/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@Variable_13
�
Variable_13/Adam_1
VariableV2*
_output_shapes
:*
dtype0*
	container *
shared_name *
shape:*
_class
loc:@Variable_13
�
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
_class
loc:@Variable_13*
validate_shape(
|
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_13
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
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable*&
_output_shapes
:dd
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_1*
_output_shapes
:d
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_2*'
_output_shapes
:d�
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_3*
_output_shapes	
:�
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_4*&
_output_shapes
:dd
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_5*
_output_shapes
:d
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_6*'
_output_shapes
:d�
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_7*
_output_shapes	
:�
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_8*&
_output_shapes
:dd
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_9*
_output_shapes
:d
�
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_10*'
_output_shapes
:d�
�
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_11*
_output_shapes	
:�
�
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_12*
_output_shapes
:	�
�
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_13*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_output_shapes
: *
use_locking( *
_class
loc:@Variable*
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_output_shapes
: *
use_locking( *
_class
loc:@Variable*
validate_shape(
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*#
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
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:���������*
output_type0	
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
Const_8Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_8*
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
: "�9�HT     6�o�	�Q_�֖�AJ��
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514��
l
xPlaceholder*
dtype0*+
_output_shapes
:���������d* 
shape:���������d
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*%
valueB"����   d      *
_output_shapes
:
l
ReshapeReshapexReshape/shape*
T0*/
_output_shapes
:���������d*
Tshape0
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
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
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable/AssignAssignVariabletruncated_normal*
T0*
_class
loc:@Variable*
use_locking(*&
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
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
use_locking(*
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
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

|
BiasAddBiasAddConv2DVariable_1/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      d   �   *
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:d�
�

Variable_2
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
_class
loc:@Variable_2*
use_locking(*'
_output_shapes
:d�*
validate_shape(
x
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*'
_output_shapes
:d�
V
Const_1Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_3
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes	
:�*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
Conv2D_1Conv2DReluVariable_2/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:����������
�
MaxPoolMaxPoolRelu_1*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   d      d   *
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
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
:dd
�

Variable_4
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
_class
loc:@Variable_4*
use_locking(*&
_output_shapes
:dd*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
T
Const_2Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_5
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
_class
loc:@Variable_5*
use_locking(*
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
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
q
truncated_normal_3/shapeConst*
dtype0*%
valueB"      d   �   *
_output_shapes
:
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
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*'
_output_shapes
:d�
�

Variable_6
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
_class
loc:@Variable_6*
use_locking(*'
_output_shapes
:d�*
validate_shape(
x
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*'
_output_shapes
:d�
V
Const_3Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
x

Variable_7
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes	
:�*
validate_shape(
l
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes	
:�
�
Conv2D_3Conv2DRelu_2Variable_6/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:����������
�
	MaxPool_1MaxPoolRelu_3*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

q
truncated_normal_4/shapeConst*
dtype0*%
valueB"   d      d   *
_output_shapes
:
\
truncated_normal_4/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_4/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
T0*
dtype0*

seed *&
_output_shapes
:dd*
seed2 
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*&
_output_shapes
:dd
{
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*&
_output_shapes
:dd
�

Variable_8
VariableV2*
shared_name *
shape:dd*&
_output_shapes
:dd*
	container *
dtype0
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*
_class
loc:@Variable_8*
use_locking(*&
_output_shapes
:dd*
validate_shape(
w
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*&
_output_shapes
:dd
T
Const_4Const*
dtype0*
valueBd*���=*
_output_shapes
:d
v

Variable_9
VariableV2*
shared_name *
shape:d*
_output_shapes
:d*
	container *
dtype0
�
Variable_9/AssignAssign
Variable_9Const_4*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:d*
validate_shape(
k
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes
:d
�
Conv2D_4Conv2DReshapeVariable_8/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*/
_output_shapes
:���������d*
strides

�
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
T0*/
_output_shapes
:���������d*
data_formatNHWC
S
Relu_4Relu	BiasAdd_4*
T0*/
_output_shapes
:���������d
q
truncated_normal_5/shapeConst*
dtype0*%
valueB"      d   �   *
_output_shapes
:
\
truncated_normal_5/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_5/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
T0*
dtype0*

seed *'
_output_shapes
:d�*
seed2 
�
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*'
_output_shapes
:d�
|
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*'
_output_shapes
:d�
�
Variable_10
VariableV2*
shared_name *
shape:d�*'
_output_shapes
:d�*
	container *
dtype0
�
Variable_10/AssignAssignVariable_10truncated_normal_5*
T0*
_class
loc:@Variable_10*
use_locking(*'
_output_shapes
:d�*
validate_shape(
{
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*'
_output_shapes
:d�
V
Const_5Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
y
Variable_11
VariableV2*
shared_name *
shape:�*
_output_shapes	
:�*
	container *
dtype0
�
Variable_11/AssignAssignVariable_11Const_5*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes	
:�*
validate_shape(
o
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*
_output_shapes	
:�
�
Conv2D_5Conv2DRelu_4Variable_10/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
	BiasAdd_5BiasAddConv2D_5Variable_11/read*
T0*0
_output_shapes
:����������*
data_formatNHWC
T
Relu_5Relu	BiasAdd_5*
T0*0
_output_shapes
:����������
�
	MaxPool_2MaxPoolRelu_5*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
T0*

Tidx0*
N*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
dtype0*
valueB"����X  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*(
_output_shapes
:����������*
Tshape0
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
T0*
out_type0*
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
T0*
dtype0*

seed *(
_output_shapes
:����������*
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
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
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
:����������
i
truncated_normal_6/shapeConst*
dtype0*
valueB"X     *
_output_shapes
:
\
truncated_normal_6/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_6/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
T0*
dtype0*

seed *
_output_shapes
:	�*
seed2 
�
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*
_output_shapes
:	�
�
Variable_12
VariableV2*
shared_name *
shape:	�*
_output_shapes
:	�*
	container *
dtype0
�
Variable_12/AssignAssignVariable_12truncated_normal_6*
T0*
_class
loc:@Variable_12*
use_locking(*
_output_shapes
:	�*
validate_shape(
s
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12*
_output_shapes
:	�
T
Const_6Const*
dtype0*
valueB*���=*
_output_shapes
:
w
Variable_13
VariableV2*
shared_name *
shape:*
_output_shapes
:*
	container *
dtype0
�
Variable_13/AssignAssignVariable_13Const_6*
T0*
_class
loc:@Variable_13*
use_locking(*
_output_shapes
:*
validate_shape(
n
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*
_output_shapes
:
�
MatMulMatMuldropout/mulVariable_12/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
V
addAddMatMulVariable_13/read*
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
ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
J
Shape_1Shapeadd*
T0*
out_type0*
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
T0*

axis *
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
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
T0*
out_type0*
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
T0*

axis *
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
T0*

axis *
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
T0*#
_output_shapes
:���������*
Tshape0
Q
Const_7Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_7*
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
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
T0*
out_type0*
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
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
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
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
a
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_12/read*
T0*
transpose_a( *(
_output_shapes
:����������*
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	�*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	�
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:���������
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
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
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
T0*
out_type0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
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
T0*(
_output_shapes
:����������*
Tshape0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:����������
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
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
_output_shapes
:
d
gradients/Reshape_1_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*0
_output_shapes
:����������*
Tshape0
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
gradients/concat_grad/ShapeShapeMaxPool*
T0*
out_type0*
_output_shapes
:
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
T0*
out_type0*
N*&
_output_shapes
:::
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
T0*.
_class$
" loc:@gradients/concat_grad/Slice*0
_output_shapes
:����������
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1*0
_output_shapes
:����������
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_2*0
_output_shapes
:����������
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradRelu_1MaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_5	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
data_formatNHWC*
paddingVALID*
T0*0
_output_shapes
:����������*
strides

�
gradients/Relu_1_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:����������
�
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:����������
�
gradients/Relu_5_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_5*
T0*0
_output_shapes
:����������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*0
_output_shapes
:����������
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:�
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*0
_output_shapes
:����������
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:�
�
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradgradients/Relu_5_grad/ReluGrad*
T0*
_output_shapes	
:�*
data_formatNHWC
y
)gradients/BiasAdd_5_grad/tuple/group_depsNoOp^gradients/Relu_5_grad/ReluGrad%^gradients/BiasAdd_5_grad/BiasAddGrad
�
1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentitygradients/Relu_5_grad/ReluGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_5_grad/ReluGrad*0
_output_shapes
:����������
�
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:d�
�
gradients/Conv2D_3_grad/ShapeNShapeNRelu_2Variable_6/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2 gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*'
_output_shapes
:d�
�
gradients/Conv2D_5_grad/ShapeNShapeNRelu_4Variable_10/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/read1gradients/BiasAdd_5_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4 gradients/Conv2D_5_grad/ShapeN:11gradients/BiasAdd_5_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter*'
_output_shapes
:d�
�
gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������d
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:���������d
�
gradients/Relu_4_grad/ReluGradReluGrad0gradients/Conv2D_5_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:���������d
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
:���������d
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
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
:���������d
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:d
�
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
T0*
_output_shapes
:d*
data_formatNHWC
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
�
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*/
_output_shapes
:���������d
�
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:d
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
�
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read*
T0*
out_type0*
N* 
_output_shapes
::
�
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
T0*J
_output_shapes8
6:4������������������������������������*
strides

�
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*/
_output_shapes
:���������d
�
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter*&
_output_shapes
:dd
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
loc:@Variable*
	container *
shared_name *
shape: *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@Variable*
use_locking(*
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
loc:@Variable*
	container *
shared_name *
shape: *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
loc:@Variable*
use_locking(*
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
Variable/Adam/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable*&
_output_shapes
:dd
�
Variable/Adam
VariableV2*
dtype0*
_class
loc:@Variable*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
_class
loc:@Variable*
use_locking(*&
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
!Variable/Adam_1/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable*&
_output_shapes
:dd
�
Variable/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
use_locking(*&
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
dtype0*
valueBd*    *
_class
loc:@Variable_1*
_output_shapes
:d
�
Variable_1/Adam
VariableV2*
dtype0*
_class
loc:@Variable_1*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_1*
use_locking(*
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
dtype0*
valueBd*    *
_class
loc:@Variable_1*
_output_shapes
:d
�
Variable_1/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_1*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_1*
use_locking(*
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
dtype0*&
valueBd�*    *
_class
loc:@Variable_2*'
_output_shapes
:d�
�
Variable_2/Adam
VariableV2*
dtype0*
_class
loc:@Variable_2*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_2*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*'
_output_shapes
:d�
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable_2*'
_output_shapes
:d�
�
Variable_2/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_2*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*'
_output_shapes
:d�
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam
VariableV2*
dtype0*
_class
loc:@Variable_3*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes	
:�*
validate_shape(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_3*
_output_shapes	
:�
�
Variable_3/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_3*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes	
:�*
validate_shape(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable_4*&
_output_shapes
:dd
�
Variable_4/Adam
VariableV2*
dtype0*
_class
loc:@Variable_4*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_4*
use_locking(*&
_output_shapes
:dd*
validate_shape(
�
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable_4*&
_output_shapes
:dd
�
Variable_4/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_4*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_4*
use_locking(*&
_output_shapes
:dd*
validate_shape(
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_class
loc:@Variable_5*
_output_shapes
:d
�
Variable_5/Adam
VariableV2*
dtype0*
_class
loc:@Variable_5*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_5*
use_locking(*
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
dtype0*
valueBd*    *
_class
loc:@Variable_5*
_output_shapes
:d
�
Variable_5/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_5*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_5*
use_locking(*
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
dtype0*&
valueBd�*    *
_class
loc:@Variable_6*'
_output_shapes
:d�
�
Variable_6/Adam
VariableV2*
dtype0*
_class
loc:@Variable_6*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_6*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*'
_output_shapes
:d�
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable_6*'
_output_shapes
:d�
�
Variable_6/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_6*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_6*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*'
_output_shapes
:d�
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_7*
_output_shapes	
:�
�
Variable_7/Adam
VariableV2*
dtype0*
_class
loc:@Variable_7*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes	
:�*
validate_shape(
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes	
:�
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_7*
_output_shapes	
:�
�
Variable_7/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_7*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes	
:�*
validate_shape(
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
_output_shapes	
:�
�
!Variable_8/Adam/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable_8*&
_output_shapes
:dd
�
Variable_8/Adam
VariableV2*
dtype0*
_class
loc:@Variable_8*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_8*
use_locking(*&
_output_shapes
:dd*
validate_shape(
�
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8*&
_output_shapes
:dd
�
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*%
valueBdd*    *
_class
loc:@Variable_8*&
_output_shapes
:dd
�
Variable_8/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_8*
	container *
shared_name *
shape:dd*&
_output_shapes
:dd
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_8*
use_locking(*&
_output_shapes
:dd*
validate_shape(
�
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8*&
_output_shapes
:dd
�
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_class
loc:@Variable_9*
_output_shapes
:d
�
Variable_9/Adam
VariableV2*
dtype0*
_class
loc:@Variable_9*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:d*
validate_shape(
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9*
_output_shapes
:d
�
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_class
loc:@Variable_9*
_output_shapes
:d
�
Variable_9/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_9*
	container *
shared_name *
shape:d*
_output_shapes
:d
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:d*
validate_shape(
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*
_output_shapes
:d
�
"Variable_10/Adam/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable_10*'
_output_shapes
:d�
�
Variable_10/Adam
VariableV2*
dtype0*
_class
loc:@Variable_10*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_10*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10*'
_output_shapes
:d�
�
$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd�*    *
_class
loc:@Variable_10*'
_output_shapes
:d�
�
Variable_10/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_10*
	container *
shared_name *
shape:d�*'
_output_shapes
:d�
�
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_10*
use_locking(*'
_output_shapes
:d�*
validate_shape(
�
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10*'
_output_shapes
:d�
�
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_11*
_output_shapes	
:�
�
Variable_11/Adam
VariableV2*
dtype0*
_class
loc:@Variable_11*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes	
:�*
validate_shape(
y
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*
_output_shapes	
:�
�
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
valueB�*    *
_class
loc:@Variable_11*
_output_shapes	
:�
�
Variable_11/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_11*
	container *
shared_name *
shape:�*
_output_shapes	
:�
�
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes	
:�*
validate_shape(
}
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11*
_output_shapes	
:�
�
"Variable_12/Adam/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_12*
_output_shapes
:	�
�
Variable_12/Adam
VariableV2*
dtype0*
_class
loc:@Variable_12*
	container *
shared_name *
shape:	�*
_output_shapes
:	�
�
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_12*
use_locking(*
_output_shapes
:	�*
validate_shape(
}
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_class
loc:@Variable_12*
_output_shapes
:	�
�
$Variable_12/Adam_1/Initializer/zerosConst*
dtype0*
valueB	�*    *
_class
loc:@Variable_12*
_output_shapes
:	�
�
Variable_12/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_12*
	container *
shared_name *
shape:	�*
_output_shapes
:	�
�
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_12*
use_locking(*
_output_shapes
:	�*
validate_shape(
�
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12*
_output_shapes
:	�
�
"Variable_13/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_13*
_output_shapes
:
�
Variable_13/Adam
VariableV2*
dtype0*
_class
loc:@Variable_13*
	container *
shared_name *
shape:*
_output_shapes
:
�
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_13*
use_locking(*
_output_shapes
:*
validate_shape(
x
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_class
loc:@Variable_13*
_output_shapes
:
�
$Variable_13/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_13*
_output_shapes
:
�
Variable_13/Adam_1
VariableV2*
dtype0*
_class
loc:@Variable_13*
	container *
shared_name *
shape:*
_output_shapes
:
�
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_13*
use_locking(*
_output_shapes
:*
validate_shape(
|
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_class
loc:@Variable_13*
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
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable*&
_output_shapes
:dd
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_1*
_output_shapes
:d
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_2*'
_output_shapes
:d�
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_3*
_output_shapes	
:�
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_4*&
_output_shapes
:dd
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_5*
_output_shapes
:d
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_6*'
_output_shapes
:d�
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_7*
_output_shapes	
:�
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_8*&
_output_shapes
:dd
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_9*
_output_shapes
:d
�
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_10*'
_output_shapes
:d�
�
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_11*
_output_shapes	
:�
�
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_12*
_output_shapes
:	�
�
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_class
loc:@Variable_13*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@Variable*
use_locking( *
_output_shapes
: *
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
use_locking( *
_output_shapes
: *
validate_shape(
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*#
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
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*#
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
Const_8Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_8*
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
: ""�!
	variables�!�!
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
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0
M
Variable_10:0Variable_10/AssignVariable_10/read:02truncated_normal_5:0
B
Variable_11:0Variable_11/AssignVariable_11/read:02	Const_5:0
M
Variable_12:0Variable_12/AssignVariable_12/read:02truncated_normal_6:0
B
Variable_13:0Variable_13/AssignVariable_13/read:02	Const_6:0
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0
h
Variable_8/Adam:0Variable_8/Adam/AssignVariable_8/Adam/read:02#Variable_8/Adam/Initializer/zeros:0
p
Variable_8/Adam_1:0Variable_8/Adam_1/AssignVariable_8/Adam_1/read:02%Variable_8/Adam_1/Initializer/zeros:0
h
Variable_9/Adam:0Variable_9/Adam/AssignVariable_9/Adam/read:02#Variable_9/Adam/Initializer/zeros:0
p
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0
l
Variable_10/Adam:0Variable_10/Adam/AssignVariable_10/Adam/read:02$Variable_10/Adam/Initializer/zeros:0
t
Variable_10/Adam_1:0Variable_10/Adam_1/AssignVariable_10/Adam_1/read:02&Variable_10/Adam_1/Initializer/zeros:0
l
Variable_11/Adam:0Variable_11/Adam/AssignVariable_11/Adam/read:02$Variable_11/Adam/Initializer/zeros:0
t
Variable_11/Adam_1:0Variable_11/Adam_1/AssignVariable_11/Adam_1/read:02&Variable_11/Adam_1/Initializer/zeros:0
l
Variable_12/Adam:0Variable_12/Adam/AssignVariable_12/Adam/read:02$Variable_12/Adam/Initializer/zeros:0
t
Variable_12/Adam_1:0Variable_12/Adam_1/AssignVariable_12/Adam_1/read:02&Variable_12/Adam_1/Initializer/zeros:0
l
Variable_13/Adam:0Variable_13/Adam/AssignVariable_13/Adam/read:02$Variable_13/Adam/Initializer/zeros:0
t
Variable_13/Adam_1:0Variable_13/Adam_1/AssignVariable_13/Adam_1/read:02&Variable_13/Adam_1/Initializer/zeros:0"
train_op

Adam"�
trainable_variables��
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
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0
M
Variable_10:0Variable_10/AssignVariable_10/read:02truncated_normal_5:0
B
Variable_11:0Variable_11/AssignVariable_11/read:02	Const_5:0
M
Variable_12:0Variable_12/AssignVariable_12/read:02truncated_normal_6:0
B
Variable_13:0Variable_13/AssignVariable_13/read:02	Const_6:0".
	summaries!

cross_entropy:0
accuracy_1:0���i4       ^3\	U���֖�A*)

cross_entropy�?


accuracy_1�?�D�6       OW��	_px�֖�A2*)

cross_entropy�+�?


accuracy_1���>akA�6       OW��	<���֖�Ad*)

cross_entropyNk�?


accuracy_1��(?�?�O7       ���Y	��q�֖�A�*)

cross_entropy|��?


accuracy_1   ?�?��7       ���Y	���֖�A�*)

cross_entropy���?


accuracy_1��?߷7       ���Y	��b�֖�A�*)

cross_entropyV�G?


accuracy_1�G?�фY7       ���Y	=���֖�A�*)

cross_entropy�n?


accuracy_1�Q8?W��O7       ���Y	6���֖�A�*)

cross_entropyI��?


accuracy_1��(?n2e�7       ���Y	P_�֖�A�*)

cross_entropy��?


accuracy_1��Q?��7       ���Y	v���֖�A�*)

cross_entropyJ��>


accuracy_1�p=?3	ލ7       ���Y	��`�֖�A�*)

cross_entropyi$S?


accuracy_1�p=?�c7       ���Y	i%�֖�A�*)

cross_entropy�,O?


accuracy_1�Q8?C��~7       ���Y	����֖�A�*)

cross_entropy#�"?


accuracy_1�G?�+<7       ���Y	8^�֖�A�*)

cross_entropy�f?


accuracy_1{.?bzG�7       ���Y	\���֖�A�*)

cross_entropy�^(?


accuracy_1�p=?��F�7       ���Y	y��֖�A�*)

cross_entropy�G>?


accuracy_1\�B?�ҏ-7       ���Y	�:�֖�A�*)

cross_entropy�*?


accuracy_1�G?r�7       ���Y	$���֖�A�*)

cross_entropya��>


accuracy_1��L?�jy@7       ���Y	�nk�֖�A�*)

cross_entropy��
?


accuracy_1�p=?��;7       ���Y	Q�֖�A�*)

cross_entropy=��>


accuracy_1��L?"0�|7       ���Y	�A��֖�A�*)

cross_entropyz�i?


accuracy_1\�B?�d�7       ���Y	�k�֖�A�*)

cross_entropy\�?


accuracy_1\�B?s�$d7       ���Y	���֖�A�*)

cross_entropy3?


accuracy_1�Q8?�nǘ7       ���Y	����֖�A�*)

cross_entropy��7?


accuracy_1333?���d7       ���Y	��D�֖�A�	*)

cross_entropy��>


accuracy_1�G?0��$7       ���Y	����֖�A�	*)

cross_entropy�]?


accuracy_1��L?�d�7       ���Y	�t�֖�A�
*)

cross_entropyzr�>


accuracy_1��L?��Ql7       ���Y	�J	�֖�A�
*)

cross_entropy��>


accuracy_1�Ga?;l��7       ���Y	�?��֖�A�
*)

cross_entropyo2�>


accuracy_1=
W?a"�87       ���Y	x9�֖�A�*)

cross_entropyO��>


accuracy_1�G?�s��7       ���Y	���֖�A�*)

cross_entropyX��>


accuracy_1�(\?L��7       ���Y	�v��֖�A�*)

cross_entropy���>


accuracy_1��L?���7       ���Y	o��֖�A�*)

cross_entropy��>


accuracy_1=
W?Z$x7       ���Y	�O��֖�A�*)

cross_entropy�g�>


accuracy_1=
W?��7       ���Y	��6�֖�A�*)

cross_entropy���>


accuracy_1�(\?~�`�7       ���Y	
��֖�A�*)

cross_entropy��>


accuracy_1\�B?��(7       ���Y	�V�֖�A�*)

cross_entropy�^~>


accuracy_1��e?BQ��7       ���Y	�9��֖�A�*)

cross_entropyMdt>


accuracy_1�Ga?R`ߩ7       ���Y	c�t�֖�A�*)

cross_entropyx��>


accuracy_1�G?L���7       ���Y	�I�֖�A�*)

cross_entropyAP�>


accuracy_1�G?%xp7       ���Y	����֖�A�*)

cross_entropy�S�>


accuracy_1�Ga?�z&�7       ���Y	��G�֖�A�*)

cross_entropy\^�>


accuracy_1��Q?����7       ���Y	,���֖�A�*)

cross_entropy�Y�>


accuracy_1��Q?���P7       ���Y	x ��֖�A�*)

cross_entropy��T>


accuracy_1�k?��C7       ���Y	��5�֖�A�*)

cross_entropy�b�>


accuracy_1�p=?�䅔7       ���Y	�%��֖�A�*)

cross_entropyZ��>


accuracy_1�Ga?�ã�7       ���Y	NB��֖�A�*)

cross_entropy<u�>


accuracy_1fff?��V�7       ���Y	�R%�֖�A�*)

cross_entropy5Ä>


accuracy_1�k?��&b7       ���Y	���֖�A�*)

cross_entropy��>


accuracy_1�(\?}��7       ���Y	��v�֖�A�*)

cross_entropy��>


accuracy_1�G?�|_7       ���Y	�k�֖�A�*)

cross_entropyl]>


accuracy_1ףp?�'=7       ���Y	����֖�A�*)

cross_entropyv�@>


accuracy_1�Ga?�u�7       ���Y	�א�֖�A�*)

cross_entropy�!>


accuracy_1��u?��]�7       ���Y	�i8�֖�A�*)

cross_entropy,h�>


accuracy_1=
W?��¾7       ���Y	���֖�A�*)

cross_entropy�nK>


accuracy_1��u?苒�7       ���Y	��֖�A�*)

cross_entropyHjv>


accuracy_1fff?;�7       ���Y	(6�֖�A�*)

cross_entropy�vX>


accuracy_1/�`?��7       ���Y	6���֖�A�*)

cross_entropysr>


accuracy_1fff?[.mS7       ���Y	���֖�A�*)

cross_entropy��>


accuracy_1�(\?蕍i7       ���Y	KxT�֖�A�*)

cross_entropye�>


accuracy_1�(\?ZD� 7       ���Y	v��֖�A�*)

cross_entropyqZR>


accuracy_1fff?���7       ���Y	0s��֖�A�*)

cross_entropy-h>


accuracy_1��u?�gB7       ���Y	���֖�A�*)

cross_entropy�>


accuracy_1ףp?�Xy7       ���Y	L�] ז�A�*)

cross_entropy|^�>


accuracy_1�k?9��7       ���Y	A\ז�A�*)

cross_entropys^�>


accuracy_1��L?}�'�7       ���Y	H��ז�A�*)

cross_entropy��[>


accuracy_1�k?\JƦ7       ���Y	Vv~ז�A�*)

cross_entropy�7>


accuracy_1H�z?��367       ���Y	��5ז�A�*)

cross_entropy4>s>


accuracy_1ףp?IM%�7       ���Y	�#�ז�A�*)

cross_entropya�J>


accuracy_1�k?Ƞ�7       ���Y	�o�ז�A�*)

cross_entropys" >


accuracy_1ףp?��+17       ���Y	!�<ז�A�*)

cross_entropy��8>


accuracy_1��u?�VP7       ���Y	�9ז�A�*)

cross_entropy�;>


accuracy_1ףp?�w��7       ���Y	�,�ז�A�*)

cross_entropy�>


accuracy_1H�z?d�v7       ���Y	��^ז�A�*)

cross_entropy>


accuracy_1��u?�G7       ���Y	�Iז�A�*)

cross_entropyo�>


accuracy_1  �?)�07       ���Y	�)�ז�A�*)

cross_entropy�vZ>


accuracy_1ףp?e�x7       ���Y	�:E	ז�A�*)

cross_entropy��>


accuracy_1ףp?(��7       ���Y	���	ז�A�*)

cross_entropy���=


accuracy_1H�z??jd�7       ���Y	a�|
ז�A�*)

cross_entropy:0(>


accuracy_1��u?���7       ���Y	�  ז�A�*)

cross_entropy�:>


accuracy_1�k?D��/7       ���Y	J��ז�A�*)

cross_entropy[(>


accuracy_1  �?���7       ���Y	�ύז�A�*)

cross_entropy5�>


accuracy_1H�z?����7       ���Y	I�8ז�A� *)

cross_entropy�FD>


accuracy_1��u?d-ȍ7       ���Y	��ז�A� *)

cross_entropy�`?>


accuracy_1��u?q�-7       ���Y	G��ז�A� *)

cross_entropy��%>


accuracy_1ףp?Ⱥ� 7       ���Y	ĩbז�A�!*)

cross_entropy[�1>


accuracy_1��u?���Q7       ���Y	(\
ז�A�!*)

cross_entropy-O�=


accuracy_1H�z?<V7       ���Y	J��ז�A�!*)

cross_entropy\J6>


accuracy_1ףp?�hR�7       ���Y	��Qז�A�"*)

cross_entropy��=


accuracy_1��z?�|?7       ���Y	��ז�A�"*)

cross_entropy�.�@


accuracy_1ףp?���7       ���Y	��ז�A�#*)

cross_entropyG&>


accuracy_1��u?��7       ���Y	�5�ז�A�#*)

cross_entropyeO9>


accuracy_1�Ga?�T|�7       ���Y	�}Jז�A�#*)

cross_entropy�}4>


accuracy_1��u?|(V�7       ���Y	�� ז�A�$*)

cross_entropy/>


accuracy_1��u?�@[7       ���Y	�7�ז�A�$*)

cross_entropy�	�=


accuracy_1H�z?�͑�7       ���Y	�Vpז�A�%*)

cross_entropy��+>


accuracy_1H�z?d��7       ���Y	?O$ז�A�%*)

cross_entropy�=


accuracy_1H�z?T�d�7       ���Y	 �ז�A�%*)

cross_entropy��=


accuracy_1H�z?��ž7       ���Y	G }ז�A�&*)

cross_entropyd��=


accuracy_1  �?Xvv7       ���Y	��3ז�A�&*)

cross_entropy�r�=


accuracy_1H�z?M��h7       ���Y	���ז�A�'*)

cross_entropyni�=


accuracy_1  �?���7       ���Y	�w�ז�A�'*)

cross_entropy,��=


accuracy_1H�z?���7       ���Y	�Omז�A�'*)

cross_entropy@�)>


accuracy_1ףp?Ԏ>F7       ���Y	�}	ז�A�(*)

cross_entropy]��=


accuracy_1��u?��x�7       ���Y	���ז�A�(*)

cross_entropy��=


accuracy_1H�z?k���7       ���Y	�mFז�A�)*)

cross_entropy�E�=


accuracy_1H�z?����7       ���Y	���ז�A�)*)

cross_entropya��=


accuracy_1H�z?�D	r7       ���Y	�Q�ז�A�)*)

cross_entropy�D�=


accuracy_1H�z?@�˞7       ���Y	�Z#ז�A�**)

cross_entropy��=


accuracy_1��u?��&7       ���Y	?|�ז�A�**)

cross_entropy��=


accuracy_1��u?YĠW7       ���Y	Ė` ז�A�**)

cross_entropy��=


accuracy_1H�z?�~w�7       ���Y	S!ז�A�+*)

cross_entropyL�=


accuracy_1��u?���7       ���Y	!��!ז�A�+*)

cross_entropy#�'>


accuracy_1��u?U]�+7       ���Y	�W["ז�A�,*)

cross_entropyE�>


accuracy_1ףp?{�y�7       ���Y	gQ�"ז�A�,*)

cross_entropy*Ѿ=


accuracy_1H�z?� ��7       ���Y	`L�#ז�A�,*)

cross_entropy(�>


accuracy_1ףp?��_7       ���Y	��1$ז�A�-*)

cross_entropy�¹=


accuracy_1  �?r}7       ���Y	�e�$ז�A�-*)

cross_entropy:��=


accuracy_1  �?�m��7       ���Y	I�~%ז�A�.*)

cross_entropy���=


accuracy_1H�z?����7       ���Y	ų&ז�A�.*)

cross_entropy�=


accuracy_1��u?E�7       ���Y	�k�&ז�A�.*)

cross_entropy��=


accuracy_1  �?hO��7       ���Y	��'ז�A�/*)

cross_entropy��=


accuracy_1  �?@��7       ���Y	��(ז�A�/*)

cross_entropy�҇=


accuracy_1  �?�%Y�7       ���Y	�(ז�A�0*)

cross_entropy]��=


accuracy_1��u?Tj{�7       ���Y	YcS)ז�A�0*)

cross_entropy�i�=


accuracy_1  �?�6 �7       ���Y	�K�)ז�A�0*)

cross_entropy��>


accuracy_1��u?��i7       ���Y	\f�*ז�A�1*)

cross_entropy�ۼ=


accuracy_1H�z?�2�7       ���Y	��0+ז�A�1*)

cross_entropy�D�=


accuracy_1H�z?��BX7       ���Y	��+ז�A�2*)

cross_entropyϘ=


accuracy_1H�z?���7       ���Y	}�,ז�A�2*)

cross_entropy��=


accuracy_1��u?wL��7       ���Y	�9-ז�A�2*)

cross_entropy��'=


accuracy_1  �?J�X7       ���Y	rl�-ז�A�3*)

cross_entropyК=


accuracy_1  �?��7       ���Y		��.ז�A�3*)

cross_entropy뺑>


accuracy_1ףp?�yv�7       ���Y	�6/ז�A�3*)

cross_entropy��=


accuracy_1  �?���7       ���Y	���/ז�A�4*)

cross_entropy���=


accuracy_1H�z?57       ���Y	"�m0ז�A�4*)

cross_entropyq��=


accuracy_1  �?�7�R7       ���Y	�1ז�A�5*)

cross_entropyf=#=


accuracy_1  �?GG{^7       ���Y	!~�1ז�A�5*)

cross_entropy�"O=


accuracy_1  �?��Ҕ7       ���Y	$2ז�A�5*)

cross_entropy�x=


accuracy_1  �?��ɋ7       ���Y	��33ז�A�6*)

cross_entropy�==


accuracy_1H�z?�7�7       ���Y	�3�3ז�A�6*)

cross_entropy~�=


accuracy_1ףp?y�<�7       ���Y	��4ז�A�7*)

cross_entropy�wM=


accuracy_1  �?�j7       ���Y	 �`5ז�A�7*)

cross_entropy�<�=


accuracy_1  �?����7       ���Y	��	6ז�A�7*)

cross_entropy6�=


accuracy_1H�z?H�B�7       ���Y	�R�6ז�A�8*)

cross_entropyď�=


accuracy_1  �?�p7       ���Y	9�J7ז�A�8*)

cross_entropyav=


accuracy_1  �?hZ&G7       ���Y	W��7ז�A�9*)

cross_entropyz�u=


accuracy_1  �?���7       ���Y	 ��8ז�A�9*)

cross_entropy�u�=


accuracy_1H�z?b��T7       ���Y	Ъ(9ז�A�9*)

cross_entropy�C�=


accuracy_1H�z?���7       ���Y	Ť�9ז�A�:*)

cross_entropy�=


accuracy_1H�z?��7       ���Y	��`:ז�A�:*)

cross_entropyB�=


accuracy_1  �?���V7       ���Y	,�%;ז�A�:*)

cross_entropy��p=


accuracy_1  �?���)7       ���Y	���;ז�A�;*)

cross_entropy���=


accuracy_1H�z?��	7       ���Y	��j<ז�A�;*)

cross_entropy(�=


accuracy_1H�z?��7       ���Y	G�=ז�A�<*)

cross_entropy�)�=


accuracy_1��u?�F��7       ���Y	���=ז�A�<*)

cross_entropy��R=


accuracy_1  �?'W��7       ���Y	#_h>ז�A�<*)

cross_entropyn�[=


accuracy_1  �?�7�7       ���Y	��?ז�A�=*)

cross_entropy��|=


accuracy_1  �?�|�7       ���Y	yN�?ז�A�=*)

cross_entropy4�7=


accuracy_1  �?s�v7       ���Y	]@o@ז�A�>*)

cross_entropy\�=


accuracy_1  �?����