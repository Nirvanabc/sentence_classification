       �K"	  @D���Abrain.Event:2�C7���      �*�	;GD���A"ݗ
l
xPlaceholder* 
shape:���������d*+
_output_shapes
:���������d*
dtype0
e
y_Placeholder*
shape:���������*'
_output_shapes
:���������*
dtype0
f
Reshape/shapeConst*%
valueB"����   d      *
_output_shapes
:*
dtype0
l
ReshapeReshapexReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������d
o
truncated_normal/shapeConst*%
valueB"   d      d   *
_output_shapes
:*
dtype0
Z
truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
truncated_normal/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*&
_output_shapes
:dd*
dtype0*
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
VariableV2*
	container *
shape:dd*&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
R
ConstConst*
valueBd*���=*
_output_shapes
:d*
dtype0
v

Variable_1
VariableV2*
	container *
shape:d*
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
Conv2DConv2DReshapeVariable/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
|
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
q
truncated_normal_1/shapeConst*%
valueB"   d      d   *
_output_shapes
:*
dtype0
\
truncated_normal_1/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
^
truncated_normal_1/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*&
_output_shapes
:dd*
dtype0*
seed2 
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
	container *
shape:dd*&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
T
Const_1Const*
valueBd*���=*
_output_shapes
:d*
dtype0
v

Variable_3
VariableV2*
	container *
shape:d*
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
Conv2D_1Conv2DReshapeVariable_2/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
q
truncated_normal_2/shapeConst*%
valueB"   d      d   *
_output_shapes
:*
dtype0
\
truncated_normal_2/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
^
truncated_normal_2/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*&
_output_shapes
:dd*
dtype0*
seed2 
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
	container *
shape:dd*&
_output_shapes
:dd*
dtype0*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
w
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
T
Const_2Const*
valueBd*���=*
_output_shapes
:d*
dtype0
v

Variable_5
VariableV2*
	container *
shape:d*
_output_shapes
:d*
dtype0*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
Conv2D_2Conv2DReshapeVariable_4/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
M
concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*
N*
T0*

Tidx0*0
_output_shapes
:����������
`
Reshape_1/shapeConst*
valueB"����,  *
_output_shapes
:*
dtype0
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������
N
	keep_probPlaceholder*
shape:*
_output_shapes
:*
dtype0
V
dropout/ShapeShape	Reshape_1*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
_output_shapes
: *
dtype0
_
dropout/random_uniform/maxConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
T0*(
_output_shapes
:����������*
dtype0*
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
valueB",     *
_output_shapes
:*
dtype0
\
truncated_normal_3/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
^
truncated_normal_3/stddevConst*
valueB
 *���=*
_output_shapes
: *
dtype0
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
T0*
_output_shapes
:	�*
dtype0*
seed2 
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
	container *
shape:	�*
_output_shapes
:	�*
dtype0*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_3Const*
valueB*���=*
_output_shapes
:*
dtype0
v

Variable_7
VariableV2*
	container *
shape:*
_output_shapes
:*
dtype0*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
MatMulMatMuldropout/mulVariable_6/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
U
addAddMatMulVariable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
value	B :*
_output_shapes
: *
dtype0
H
ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
J
Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*
_output_shapes
:*

axis 
T

Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
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
���������*
_output_shapes
:*
dtype0
O
concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
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
Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
I
Shape_2Shapey_*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*
T0*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_2/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
O
concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
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
Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
U
Slice_2/sizePackSub_2*
N*
T0*
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
valueB: *
_output_shapes
:*
dtype0
^
MeanMean	Reshape_4Const_4*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
`
cross_entropy/tagsConst*
valueB Bcross_entropy*
_output_shapes
: *
dtype0
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
T
gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
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
gradients/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
�
gradients/Mean_grad/ConstConst*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
dtype0
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
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
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
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
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
T0*
Tshape0*'
_output_shapes
:���������
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
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
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:����������
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
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
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0*(
_output_shapes
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
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
T0*
Tshape0*0
_output_shapes
:����������
\
gradients/concat_grad/RankConst*
value	B :*
_output_shapes
: *
dtype0
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
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*.
_class$
" loc:@gradients/concat_grad/Slice*
T0*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0*/
_output_shapes
:���������d
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
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
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:d
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
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
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
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
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
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
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
_output_shapes
: *
dtype0
�
beta1_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*
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
beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Variable*
_output_shapes
: *
dtype0
�
beta2_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*
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
Variable/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0
�
Variable/Adam
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd
{
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
�
!Variable/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0
�
Variable/Adam_1
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
�
!Variable_1/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0
�
Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
#Variable_1/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0
�
Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
!Variable_2/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0
�
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
�
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0
�
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
�
!Variable_3/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0
�
Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
#Variable_3/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0
�
Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
!Variable_4/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0
�
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
�
#Variable_4/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0
�
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
�
!Variable_5/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0
�
Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
#Variable_5/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0
�
Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
!Variable_6/Adam/Initializer/zerosConst*
valueB	�*    *
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0
�
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
valueB	�*    *
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0
�
Variable_6/Adam_1
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
_output_shapes
:*
dtype0
�
Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
_output_shapes
:*
dtype0
�
Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *��8*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
valueB
 *w�?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*
T0*

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
Const_5Const*
valueB: *
_output_shapes
:*
dtype0
_
accuracyMeanCast_1Const_5*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
_output_shapes
: *
dtype0
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
: "v����      eef�	�1WD���AJ��
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
shape:���������d*
dtype0*+
_output_shapes
:���������d
e
y_Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
f
Reshape/shapeConst*%
valueB"����   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*
T0*/
_output_shapes
:���������d
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
dtype0*&
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
VariableV2*
	container *
shape:dd*
dtype0*&
_output_shapes
:dd*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
R
ConstConst*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_1
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
Conv2DConv2DReshapeVariable/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
|
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:���������d
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*&
_output_shapes
:dd*
seed2 
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
	container *
shape:dd*
dtype0*&
_output_shapes
:dd*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
T
Const_1Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_3
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
Conv2D_1Conv2DReshapeVariable_2/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
S
Relu_1Relu	BiasAdd_1*
T0*/
_output_shapes
:���������d
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
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
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*
dtype0*&
_output_shapes
:dd*
seed2 
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
	container *
shape:dd*
dtype0*&
_output_shapes
:dd*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
w
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
T
Const_2Const*
valueBd*���=*
dtype0*
_output_shapes
:d
v

Variable_5
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
Conv2D_2Conv2DReshapeVariable_4/read*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*
T0*/
_output_shapes
:���������d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:���������d
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
M
concat/axisConst*
value	B :*
dtype0*
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
valueB"����,  *
dtype0*
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
Tshape0*
T0*(
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
T0*
out_type0*
_output_shapes
:
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
T0*
dtype0*(
_output_shapes
:����������*
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
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
T0*
dtype0*
_output_shapes
:	�*
seed2 
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
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_3Const*
valueB*���=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
MatMulMatMuldropout/mulVariable_6/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:���������
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
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
J
Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
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
N*
T0*

axis *
_output_shapes
:
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
N*
T0*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
Tshape0*
T0*0
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
T0*
out_type0*
_output_shapes
:
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
N*
T0*

axis *
_output_shapes
:
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
N*
T0*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
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
N*
T0*

axis *
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
Const_4Const*
valueB: *
dtype0*
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*

Tidx0*
_output_shapes
: 
�
gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
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
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
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
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
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
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:����������
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
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
:����������
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
_output_shapes
:
d
gradients/Reshape_1_grad/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
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
T0*
out_type0*
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
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*.
_class$
" loc:@gradients/concat_grad/Slice*
T0*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0*/
_output_shapes
:���������d
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0*/
_output_shapes
:���������d
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
data_formatNHWC*
strides
*/
_output_shapes
:���������d*
ksize
*
paddingVALID*
T0
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
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:d
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:d
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0*/
_output_shapes
:���������d
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
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
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
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
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
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
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
data_formatNHWC*
strides
*J
_output_shapes8
6:4������������������������������������*
use_cudnn_on_gpu(*
paddingVALID*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������d
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*
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
beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*
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
Variable/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable*
dtype0*&
_output_shapes
:dd
�
Variable/Adam
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd
{
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
�
!Variable/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable*
dtype0*&
_output_shapes
:dd
�
Variable/Adam_1
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(*&
_output_shapes
:dd

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
�
!Variable_1/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:d
�
Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
#Variable_1/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:d
�
Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
�
!Variable_2/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_2*
dtype0*&
_output_shapes
:dd
�
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
�
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_2*
dtype0*&
_output_shapes
:dd
�
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*&
_output_shapes
:dd
�
!Variable_3/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:d
�
Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
#Variable_3/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:d
�
Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
!Variable_4/Adam/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_4*
dtype0*&
_output_shapes
:dd
�
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
�
#Variable_4/Adam_1/Initializer/zerosConst*%
valueBdd*    *
_class
loc:@Variable_4*
dtype0*&
_output_shapes
:dd
�
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*&
_output_shapes
:dd*
dtype0*
shared_name *
	container *
shape:dd
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(*&
_output_shapes
:dd
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*&
_output_shapes
:dd
�
!Variable_5/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:d
�
Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
#Variable_5/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:d
�
Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
_output_shapes
:d*
dtype0*
shared_name *
	container *
shape:d
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(*
_output_shapes
:d
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes
:d
�
!Variable_6/Adam/Initializer/zerosConst*
valueB	�*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:	�
�
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/zerosConst*
valueB	�*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:	�
�
Variable_6/Adam_1
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	�*
dtype0*
shared_name *
	container *
shape:	�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(*
_output_shapes
:	�
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:
�
Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shared_name *
	container *
shape:
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(*
_output_shapes
:
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:
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
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable*&
_output_shapes
:dd
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_1*
_output_shapes
:d
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_2*&
_output_shapes
:dd
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_3*
_output_shapes
:d
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_4*&
_output_shapes
:dd
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_5*
_output_shapes
:d
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_6*
_output_shapes
:	�
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*
_class
loc:@Variable_7*
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*
T0*

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
Const_5Const*
valueB: *
dtype0*
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
: "".
	summaries!

cross_entropy:0
accuracy_1:0"�
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
Adam��_�4       ^3\	�{�D���A*)

cross_entropy_@


accuracy_1q=
?_��6       OW��	���D���A2*)

cross_entropyل@


accuracy_1���>_f!X6       OW��	˹�D���Ad*)

cross_entropy&�?


accuracy_1�z?\<�]7       ���Y	ZE���A�*)

cross_entropy�F�?


accuracy_1�Q8?Gx�7       ���Y	�6DE���A�*)

cross_entropy!�C?


accuracy_1
�#?Ɋs�7       ���Y	��jE���A�*)

cross_entropy��V?


accuracy_1��(?�W0:7       ���Y	�؏E���A�*)

cross_entropy���?


accuracy_1�?���7       ���Y	���E���A�*)

cross_entropy<��?


accuracy_1��(?��r7       ���Y	<�E���A�*)

cross_entropy��k?


accuracy_1�Q8?����7       ���Y	�F���A�*)

cross_entropy{٠?


accuracy_1�?D��n7       ���Y	�(F���A�*)

cross_entropy<P�?


accuracy_1�?� �:7       ���Y	�W[F���A�*)

cross_entropy(!�?


accuracy_1R�?�-P�7       ���Y	4�F���A�*)

cross_entropy`џ?


accuracy_1R�?J���7       ���Y	,�F���A�*)

cross_entropy�/�?


accuracy_1)\?2|%7       ���Y	Ȼ�F���A�*)

cross_entropy� z?


accuracy_1�p=?�/7       ���Y	���F���A�*)

cross_entropy���?


accuracy_1{.?q��7       ���Y	 CG���A�*)

cross_entropyp�?


accuracy_1��?z Ψ7       ���Y	#>G���A�*)

cross_entropy�/�?


accuracy_1�Q8?�n�7       ���Y	�dG���A�*)

cross_entropyV�?


accuracy_1�z?W<� 7       ���Y	�G���A�*)

cross_entropyc�k?


accuracy_1R�?q�x7       ���Y	�G���A�*)

cross_entropy�.l?


accuracy_1R�?a�"-7       ���Y	e��G���A�*)

cross_entropy��]?


accuracy_1�Q8?.�#7       ���Y	�H���A�*)

cross_entropy-E+?


accuracy_1�p=?�f8>7       ���Y	�4H���A�*)

cross_entropy�t?


accuracy_1��(?D��7       ���Y	@)ZH���A�	*)

cross_entropy� L?


accuracy_1333?8g�67       ���Y	�H���A�	*)

cross_entropy���?


accuracy_1
�#?�
ڿ7       ���Y	"
�H���A�
*)

cross_entropyv_2?


accuracy_1�Q8?x�r�7       ���Y		��H���A�
*)

cross_entropy{�c?


accuracy_1R�?�s�|7       ���Y	��H���A�
*)

cross_entropyl�?


accuracy_1\�B?�5��7       ���Y	�[I���A�*)

cross_entropy!5?


accuracy_1\�B?��7       ���Y	�=I���A�*)

cross_entropyO�?


accuracy_1�Q8?�}��7       ���Y	׬uI���A�*)

cross_entropy��?


accuracy_1�p=?vE�N7       ���Y	H�I���A�*)

cross_entropy)??


accuracy_1\�B?k�I7       ���Y	Rz�I���A�*)

cross_entropy��?


accuracy_1�p=?[:�>7       ���Y	%��I���A�*)

cross_entropyw/?


accuracy_1�p=?"���7       ���Y	�vJ���A�*)

cross_entropy�"8?


accuracy_1�Q8?�%.7       ���Y	�<3J���A�*)

cross_entropy,�J?


accuracy_1�<?�?@�7       ���Y	GYJ���A�*)

cross_entropy{�>?


accuracy_1{.?ӟrv7       ���Y	�|~J���A�*)

cross_entropyTR?


accuracy_1\�B?%V��7       ���Y	]�J���A�*)

cross_entropy���>


accuracy_1�G?^#��7       ���Y	���J���A�*)

cross_entropy��7?


accuracy_1�Q8?o/?w7       ���Y	�K���A�*)

cross_entropy�o-?


accuracy_1�p=?����7       ���Y	�*K���A�*)

cross_entropy�6X?


accuracy_1{.?���7       ���Y	t�OK���A�*)

cross_entropy�?


accuracy_1��(?��67       ���Y		=vK���A�*)

cross_entropy��W?


accuracy_1333?�Ap7       ���Y	�7�K���A�*)

cross_entropy�U�>


accuracy_1�Ga?Q�7       ���Y	��K���A�*)

cross_entropy�C?


accuracy_1\�B?�"7       ���Y	2G�K���A�*)

cross_entropy5_?


accuracy_1�(\?T{��7       ���Y	�3L���A�*)

cross_entropy�?


accuracy_1�Q8?���7       ���Y	�_4L���A�*)

cross_entropyz?


accuracy_1��(?��Ӗ7       ���Y	�TZL���A�*)

cross_entropyګ?


accuracy_1�p=?<K3O7       ���Y	mu�L���A�*)

cross_entropy��?


accuracy_1��L?;�l7       ���Y	���L���A�*)

cross_entropyݹ?


accuracy_1�Q8?��oy7       ���Y	{�L���A�*)

cross_entropy�{L?


accuracy_1{.?���h7       ���Y	�9�L���A�*)

cross_entropyq$?


accuracy_1�Q8?�]�7       ���Y	EN$M���A�*)

cross_entropy�3T?


accuracy_1
�#?�x�7       ���Y	��IM���A�*)

cross_entropy�R>?


accuracy_1��!?YJ�'7       ���Y	!SpM���A�*)

cross_entropy�Ɋ>


accuracy_1�Ga?R�6/7       ���Y	Z�M���A�*)

cross_entropy:?


accuracy_1�p=?L��7       ���Y	sֻM���A�*)

cross_entropy��?


accuracy_1�G?>���7       ���Y	��M���A�*)

cross_entropy�T�>


accuracy_1�Q8?�Ͽ�7       ���Y	�hN���A�*)

cross_entropy_�?


accuracy_1�Q8?��".7       ���Y	�CAN���A�*)

cross_entropyd��>


accuracy_1��L?�}B/7       ���Y	_1gN���A�*)

cross_entropyZ&?


accuracy_1�G?�)��7       ���Y	h�N���A�*)

cross_entropy�4%?


accuracy_1333?�ܨ?7       ���Y	&b�N���A�*)

cross_entropy@�W?


accuracy_1R�?�m�y7       ���Y	*k�N���A�*)

cross_entropyt,-?


accuracy_1333?�N|L7       ���Y	h�N���A�*)

cross_entropy�?


accuracy_1\�B?�6K17       ���Y	��%O���A�*)

cross_entropy� ?


accuracy_1��L? o�7       ���Y	��KO���A�*)

cross_entropy�'�>


accuracy_1�G?N�]�7       ���Y	e�qO���A�*)

cross_entropyQ�?


accuracy_1\�B?��7       ���Y	$�O���A�*)

cross_entropy�#	?


accuracy_1�G?W� 7       ���Y	��O���A�*)

cross_entropy��>


accuracy_1=
W?�]�O7       ���Y	n��O���A�*)

cross_entropyV��>


accuracy_1��L?�F�7       ���Y	l;P���A�*)

cross_entropy��?


accuracy_1�p=?���7       ���Y	��AP���A�*)

cross_entropy��#?


accuracy_1�G?YL�7       ���Y	z�gP���A�*)

cross_entropy
*?


accuracy_1�Q8?��\7       ���Y	T�P���A�*)

cross_entropy�(�>


accuracy_1fff?1�k7       ���Y	D��P���A�*)

cross_entropy���>


accuracy_1��L?4��7       ���Y	��P���A�*)

cross_entropy<�>


accuracy_1�p=?K!7       ���Y	�4�P���A�*)

cross_entropy��>


accuracy_1�(\?�3[7       ���Y	��8Q���A�*)

cross_entropy�*(?


accuracy_1\�B?.��7       ���Y	8:^Q���A� *)

cross_entropy�z�>


accuracy_1�Ga?;}�7       ���Y	VɃQ���A� *)

cross_entropy��>


accuracy_1�(\?��>7       ���Y	m˩Q���A� *)

cross_entropyS��>


accuracy_1�(\?͊*7       ���Y	�K�Q���A�!*)

cross_entropyR��>


accuracy_1��Q?x'��7       ���Y	5�Q���A�!*)

cross_entropyA��?


accuracy_1=
W?�7��7       ���Y	�0R���A�!*)

cross_entropyD�?


accuracy_1��L?��G7       ���Y	��@R���A�"*)

cross_entropy��>


accuracy_1��P?����7       ���Y	��fR���A�"*)

cross_entropy\�?


accuracy_1�G?�_�x7       ���Y	!Z�R���A�#*)

cross_entropy׫�>


accuracy_1�p=?�o�7       ���Y	k��R���A�#*)

cross_entropy���>


accuracy_1�Ga?����7       ���Y	�$�R���A�#*)

cross_entropy���>


accuracy_1�(\?�髻7       ���Y	p�
S���A�$*)

cross_entropyʿ>


accuracy_1��L?a/��7       ���Y	�1S���A�$*)

cross_entropy���>


accuracy_1�G?��Ӵ7       ���Y	�XVS���A�%*)

cross_entropy��>


accuracy_1�p=?=��7       ���Y	*P|S���A�%*)

cross_entropy�f�?


accuracy_1�G?�I87       ���Y	��S���A�%*)

cross_entropy&p�>


accuracy_1��Q?�a��7       ���Y	x�S���A�&*)

cross_entropy�@�>


accuracy_1�(\?�Y�7       ���Y	��S���A�&*)

cross_entropy���>


accuracy_1=
W?/i_E7       ���Y	.T���A�'*)

cross_entropyx̕>


accuracy_1�Ga?�U��7       ���Y	VRMT���A�'*)

cross_entropy�$�>


accuracy_1�Ga?��]	7       ���Y	�)sT���A�'*)

cross_entropy���>


accuracy_1��L?��G�7       ���Y	C �T���A�(*)

cross_entropy*��>


accuracy_1ףp?�g�[7       ���Y	t��T���A�(*)

cross_entropy�C?


accuracy_1333?�N�}7       ���Y	��T���A�)*)

cross_entropy��>


accuracy_1��L?M0�<7       ���Y	��
U���A�)*)

cross_entropy��>


accuracy_1=
W?���o7       ���Y	�1U���A�)*)

cross_entropy��>


accuracy_1��L?$M�7       ���Y	ۚVU���A�**)

cross_entropy��?


accuracy_1333?o��7       ���Y	�\|U���A�**)

cross_entropy(��>


accuracy_1�(\?�8�o7       ���Y	I�U���A�**)

cross_entropy� �>


accuracy_1�(\?��v�7       ���Y	�j�U���A�+*)

cross_entropy���>


accuracy_1=
W?��7       ���Y	XKV���A�+*)

cross_entropy��z>


accuracy_1ףp?��/�7       ���Y	4'V���A�,*)

cross_entropy.
?


accuracy_1�p=?��g@7       ���Y	l�LV���A�,*)

cross_entropy�2�>


accuracy_1�Ga?� �7       ���Y	p�rV���A�,*)

cross_entropy/��>


accuracy_1��L?T|�27       ���Y	�<�V���A�-*)

cross_entropy!Н>


accuracy_1=
W?�l2�7       ���Y	ʛ�V���A�-*)

cross_entropy���>


accuracy_1�Ga?�)"[7       ���Y	���V���A�.*)

cross_entropytJ�>


accuracy_1=
W?��t7       ���Y	|�W���A�.*)

cross_entropy��>


accuracy_1��Q?κSW7       ���Y	v/W���A�.*)

cross_entropy<`�>


accuracy_1=
W?�#�O7       ���Y	��iW���A�/*)

cross_entropy�lZ?


accuracy_1�G?$��N7       ���Y	�
�W���A�/*)

cross_entropy���>


accuracy_1��Q??�$�7       ���Y	��W���A�0*)

cross_entropyEj>


accuracy_1ףp?�!�7       ���Y	��W���A�0*)

cross_entropy�>�>


accuracy_1��Q?�0��7       ���Y	D.X���A�0*)

cross_entropy�3�>


accuracy_1=
W?8�[]7       ���Y	��'X���A�1*)

cross_entropy��>


accuracy_1=
W??<Z7       ���Y	��MX���A�1*)

cross_entropy�5�>


accuracy_1�Ga?Y�o	7       ���Y	'�tX���A�2*)

cross_entropya��>


accuracy_1�Ga?�L�7       ���Y	�ؙX���A�2*)

cross_entropy�¹>


accuracy_1��L?�2��7       ���Y	|ֿX���A�2*)

cross_entropyao>


accuracy_1�k?�T7       ���Y	SE�X���A�3*)

cross_entropy���>


accuracy_1�(\?��	7       ���Y	��Y���A�3*)

cross_entropyEe�>


accuracy_1�Q8?�H�7       ���Y	e�>Y���A�3*)

cross_entropy�½>


accuracy_1�(\?�%v7       ���Y	b�dY���A�4*)

cross_entropy+�>


accuracy_1=
W?fj�7       ���Y	��Y���A�4*)

cross_entropy3�>


accuracy_1=
W?#t�P7       ���Y	*L�Y���A�5*)

cross_entropy.2M>


accuracy_1fff?�R,7       ���Y	�4�Y���A�5*)

cross_entropy�<>


accuracy_1H�z?N
7       ���Y	��Y���A�5*)

cross_entropy���>


accuracy_1�k?-pF�7       ���Y	V�#Z���A�6*)

cross_entropy�w�>


accuracy_1=
W?���97       ���Y	�JZ���A�6*)

cross_entropy���>


accuracy_1�G?S=b�7       ���Y	��Z���A�7*)

cross_entropy��>


accuracy_1�k?��a?7       ���Y	e1�Z���A�7*)

cross_entropyQ
�>


accuracy_1��L?J�N7       ���Y	5��Z���A�7*)

cross_entropyԆ�>


accuracy_1��Q?���7       ���Y	C��Z���A�8*)

cross_entropy��v>


accuracy_1fff?)��7       ���Y	V[���A�8*)

cross_entropyt�>


accuracy_1�(\?dj�w7       ���Y	�B[���A�9*)

cross_entropy�>�>


accuracy_1�G?=Y7       ���Y	D�h[���A�9*)

cross_entropy�p�>


accuracy_1��Q?C�	M7       ���Y	bk�[���A�9*)

cross_entropy(m>


accuracy_1fff?H7       ���Y	E��[���A�:*)

cross_entropy`v�>


accuracy_1=
W?����7       ���Y	��[���A�:*)

cross_entropyc�7>


accuracy_1ףp?�� �7       ���Y	�O\���A�:*)

cross_entropy.��>


accuracy_1=
W?@��7       ���Y	�V<\���A�;*)

cross_entropy!�>


accuracy_1�Ga?+=�7       ���Y	��b\���A�;*)

cross_entropy��{>


accuracy_1�k?�ȥ7       ���Y	X݉\���A�<*)

cross_entropy�>


accuracy_1=
W?���h7       ���Y	n��\���A�<*)

cross_entropyA&�>


accuracy_1�(\?�l��7       ���Y	���\���A�<*)

cross_entropy4��>


accuracy_1�(\?ʰW}7       ���Y	C��\���A�=*)

cross_entropy���>


accuracy_1��Q?*�*7       ���Y	��#]���A�=*)

cross_entropy�Q�>


accuracy_1fff?R��7       ���Y	joJ]���A�>*)

cross_entropy�l>


accuracy_1ףp?�B�