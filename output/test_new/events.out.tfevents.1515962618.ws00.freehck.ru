       �K"	  �>��Abrain.Event:2c�3��      �6	"��>��A"��
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
:���������d*
Tshape0*
T0
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable/AssignAssignVariabletruncated_normal*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
R
ConstConst*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_1
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_1/AssignAssign
Variable_1Const*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
U
SigmoidSigmoidBiasAdd*/
_output_shapes
:���������d*
T0
�
MaxPoolMaxPoolSigmoid*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
T
Const_1Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_3
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_1Sigmoid	BiasAdd_1*/
_output_shapes
:���������d*
T0
�
	MaxPool_1MaxPool	Sigmoid_1*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
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
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
w
Variable_4/readIdentity
Variable_4*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
T
Const_2Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_5
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
�
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_2Sigmoid	BiasAdd_2*/
_output_shapes
:���������d*
T0
�
	MaxPool_2MaxPool	Sigmoid_2*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
q
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
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
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*&
_output_shapes
:dd*
T0
�

Variable_6
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
w
Variable_6/readIdentity
Variable_6*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
T
Const_3Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_7
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_3Sigmoid	BiasAdd_3*/
_output_shapes
:���������d*
T0
�
	MaxPool_3MaxPool	Sigmoid_3*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3concat/axis*0
_output_shapes
:����������*
N*

Tidx0*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
Tshape0*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
T0
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed2 *
dtype0*(
_output_shapes
:����������*
T0*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
:����������*
T0
i
truncated_normal_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     
\
truncated_normal_4/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_4/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
seed2 *
dtype0*
_output_shapes
:	�*
T0*

seed 
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	�*
T0
�

Variable_8
VariableV2*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
shape:	�
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
p
Variable_8/readIdentity
Variable_8*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_9
VariableV2*
_output_shapes
:*
dtype0*
	container *
shared_name *
shape:
�
Variable_9/AssignAssign
Variable_9Const_4*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
MatMulMatMuldropout/mulVariable_8/read*'
_output_shapes
:���������*
transpose_b( *
transpose_a( *
T0
U
addAddMatMulVariable_9/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
H
ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
J
Shape_1Shapeadd*
_output_shapes
:*
out_type0*
T0
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
Slice/beginPackSub*
_output_shapes
:*
N*

axis *
T0
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
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
_output_shapes
:*
N*

Tidx0*
T0
l
	Reshape_2Reshapeaddconcat_1*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
I
Shape_2Shapey_*
_output_shapes
:*
out_type0*
T0
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
Slice_1/beginPackSub_1*
_output_shapes
:*
N*

axis *
T0
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
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
_output_shapes
:*
N*

Tidx0*
T0
k
	Reshape_3Reshapey_concat_2*0
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
Const_5Const*
_output_shapes
:*
dtype0*
valueB: 
^
MeanMean	Reshape_4Const_5*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
`
cross_entropy/tagsConst*
_output_shapes
: *
dtype0*
valueB Bcross_entropy
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
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
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
out_type0*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*
T0*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
out_type0*
T0
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
: *

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
: *

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
out_type0*
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
a
gradients/Reshape_2_grad/ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
out_type0*
T0
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_8/read*(
_output_shapes
:����������*
transpose_b(*
transpose_a( *
T0
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_b( *
transpose_a(*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:���������*
out_type0*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*#
_output_shapes
:���������*
out_type0*
T0
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
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
T0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*#
_output_shapes
:���������*
out_type0*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
:*

Tidx0*
	keep_dims( *
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0
d
gradients/Reshape_1_grad/ShapeShapeconcat*
_output_shapes
:*
out_type0*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
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
:*
out_type0*
T0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3*,
_output_shapes
::::*
N*
T0*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*,
_output_shapes
::::*
N
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
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*.
_class$
" loc:@gradients/concat_grad/Slice*
T0
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0
�
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_3*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradSigmoidMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGrad	Sigmoid_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGrad	Sigmoid_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGrad	Sigmoid_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid"gradients/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1$gradients/MaxPool_1_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_2_grad/SigmoidGradSigmoidGrad	Sigmoid_2$gradients/MaxPool_2_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_3_grad/SigmoidGradSigmoidGrad	Sigmoid_3$gradients/MaxPool_3_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/Sigmoid_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0
y
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/Sigmoid_grad/SigmoidGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/Sigmoid_grad/SigmoidGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������d*5
_class+
)'loc:@gradients/Sigmoid_grad/SigmoidGrad*
T0
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_1_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_1_grad/tuple/group_depsNoOp%^gradients/Sigmoid_1_grad/SigmoidGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_1_grad/SigmoidGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_1_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_2_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_2_grad/tuple/group_depsNoOp%^gradients/Sigmoid_2_grad/SigmoidGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_2_grad/SigmoidGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_2_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_3_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_3_grad/tuple/group_depsNoOp%^gradients/Sigmoid_3_grad/SigmoidGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_3_grad/SigmoidGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_3_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read* 
_output_shapes
::*
N*
T0*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:dd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*
T0
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
shape: *
_output_shapes
: *
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@Variable*
T0
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
shape: *
_output_shapes
: *
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@Variable*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
�
Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
�
!Variable_6/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_6*%
valueBdd*    
�
Variable_6/Adam
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
�
Variable_6/Adam/readIdentityVariable_6/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_6*%
valueBdd*    
�
Variable_6/Adam_1
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*&
_output_shapes
:dd*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_7*
valueBd*    
�
Variable_7/Adam
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_7*
valueBd*    
�
Variable_7/Adam_1
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_8*
valueB	�*    
�
Variable_8/Adam
VariableV2*
shape:	�*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
z
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_8*
valueB	�*    
�
Variable_8/Adam_1
VariableV2*
shape:	�*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
valueB*    
�
Variable_9/Adam
VariableV2*
shape:*
_output_shapes
:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
valueB*    
�
Variable_9/Adam_1
VariableV2*
shape:*
_output_shapes
:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes
:*
_class
loc:@Variable_9*
T0
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
:dd*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_1
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_2
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_3
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_4
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_5
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_6
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_7
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_8
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *
T0*
use_locking( *
_class
loc:@Variable_9
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_output_shapes
: *
validate_shape(*
use_locking( *
_class
loc:@Variable
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_output_shapes
: *
validate_shape(*
use_locking( *
_class
loc:@Variable
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*#
_output_shapes
:���������*

DstT0*

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
: *

Tidx0*
	keep_dims( *
T0
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
accuracy_1*
_output_shapes
: *
N"=-��     �x��	2��>��AJ˥
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
/
Sigmoid
x"T
y"T"
Ttype:	
2
<
SigmoidGrad
y"T
dy"T
z"T"
Ttype:	
2
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
:���������d*
Tshape0*
T0
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
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
R
ConstConst*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_1
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_1/AssignAssign
Variable_1Const*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
|
BiasAddBiasAddConv2DVariable_1/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
U
SigmoidSigmoidBiasAdd*/
_output_shapes
:���������d*
T0
�
MaxPoolMaxPoolSigmoid*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
T
Const_1Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_3
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_3/AssignAssign
Variable_3Const_1*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_1Sigmoid	BiasAdd_1*/
_output_shapes
:���������d*
T0
�
	MaxPool_1MaxPool	Sigmoid_1*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
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
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
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
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
w
Variable_4/readIdentity
Variable_4*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
T
Const_2Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_5
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_5/AssignAssign
Variable_5Const_2*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
�
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_2Sigmoid	BiasAdd_2*/
_output_shapes
:���������d*
T0
�
	MaxPool_2MaxPool	Sigmoid_2*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
q
truncated_normal_3/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   d      d   
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
seed2 *
dtype0*&
_output_shapes
:dd*
T0*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*&
_output_shapes
:dd*
T0
{
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*&
_output_shapes
:dd*
T0
�

Variable_6
VariableV2*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
shape:dd
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
w
Variable_6/readIdentity
Variable_6*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
T
Const_3Const*
_output_shapes
:d*
dtype0*
valueBd*���=
v

Variable_7
VariableV2*
_output_shapes
:d*
dtype0*
	container *
shared_name *
shape:d
�
Variable_7/AssignAssign
Variable_7Const_3*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*/
_output_shapes
:���������d*
data_formatNHWC*
T0
Y
	Sigmoid_3Sigmoid	BiasAdd_3*/
_output_shapes
:���������d*
T0
�
	MaxPool_3MaxPool	Sigmoid_3*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3concat/axis*0
_output_shapes
:����������*
N*

Tidx0*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
Tshape0*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
T0
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
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed2 *
dtype0*(
_output_shapes
:����������*
T0*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
:����������*
T0
i
truncated_normal_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     
\
truncated_normal_4/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
^
truncated_normal_4/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
seed2 *
dtype0*
_output_shapes
:	�*
T0*

seed 
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	�*
T0
�

Variable_8
VariableV2*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
shape:	�
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
p
Variable_8/readIdentity
Variable_8*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*���=
v

Variable_9
VariableV2*
_output_shapes
:*
dtype0*
	container *
shared_name *
shape:
�
Variable_9/AssignAssign
Variable_9Const_4*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_9
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
MatMulMatMuldropout/mulVariable_8/read*'
_output_shapes
:���������*
transpose_b( *
T0*
transpose_a( 
U
addAddMatMulVariable_9/read*'
_output_shapes
:���������*
T0
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
H
ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
J
Shape_1Shapeadd*
_output_shapes
:*
out_type0*
T0
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
Slice/beginPackSub*
_output_shapes
:*
N*

axis *
T0
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
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*
_output_shapes
:*
N*

Tidx0*
T0
l
	Reshape_2Reshapeaddconcat_1*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
I
Shape_2Shapey_*
_output_shapes
:*
out_type0*
T0
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
Slice_1/beginPackSub_1*
_output_shapes
:*
N*

axis *
T0
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
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*
_output_shapes
:*
N*

Tidx0*
T0
k
	Reshape_3Reshapey_concat_2*0
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
Const_5Const*
_output_shapes
:*
dtype0*
valueB: 
^
MeanMean	Reshape_4Const_5*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
`
cross_entropy/tagsConst*
_output_shapes
: *
dtype0*
valueB Bcross_entropy
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
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
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
out_type0*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
out_type0*
T0
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
: *

Tidx0*
T0*
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
: *

Tidx0*
T0*
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
out_type0*
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
a
gradients/Reshape_2_grad/ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
out_type0*
T0
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_8/read*(
_output_shapes
:����������*
transpose_b(*
T0*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_b( *
T0*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*#
_output_shapes
:���������*
out_type0*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*#
_output_shapes
:���������*
out_type0*
T0
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
:*

Tidx0*
T0*
	keep_dims( 
�
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0
�
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0
�
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
�
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0
�
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
T0
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*#
_output_shapes
:���������*
out_type0*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
:*

Tidx0*
T0*
	keep_dims( 
�
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
�
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0
�
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0
d
gradients/Reshape_1_grad/ShapeShapeconcat*
_output_shapes
:*
out_type0*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
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
:*
out_type0*
T0
�
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3*,
_output_shapes
::::*
N*
T0*
out_type0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*,
_output_shapes
::::*
N
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
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*J
_output_shapes8
6:4������������������������������������*
T0*
Index0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*.
_class$
" loc:@gradients/concat_grad/Slice*
T0
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0
�
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*/
_output_shapes
:���������d*0
_class&
$"loc:@gradients/concat_grad/Slice_3*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradSigmoidMaxPool.gradients/concat_grad/tuple/control_dependency*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGrad	Sigmoid_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGrad	Sigmoid_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGrad	Sigmoid_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
ksize
*
strides
*
T0*/
_output_shapes
:���������d*
paddingVALID*
data_formatNHWC
�
"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid"gradients/MaxPool_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1$gradients/MaxPool_1_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_2_grad/SigmoidGradSigmoidGrad	Sigmoid_2$gradients/MaxPool_2_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
$gradients/Sigmoid_3_grad/SigmoidGradSigmoidGrad	Sigmoid_3$gradients/MaxPool_3_grad/MaxPoolGrad*/
_output_shapes
:���������d*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/Sigmoid_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0
y
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/Sigmoid_grad/SigmoidGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/Sigmoid_grad/SigmoidGrad(^gradients/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:���������d*5
_class+
)'loc:@gradients/Sigmoid_grad/SigmoidGrad*
T0
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_1_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_1_grad/tuple/group_depsNoOp%^gradients/Sigmoid_1_grad/SigmoidGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_1_grad/SigmoidGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_1_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_2_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_2_grad/tuple/group_depsNoOp%^gradients/Sigmoid_2_grad/SigmoidGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_2_grad/SigmoidGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_2_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_3_grad/SigmoidGrad*
_output_shapes
:d*
data_formatNHWC*
T0

)gradients/BiasAdd_3_grad/tuple/group_depsNoOp%^gradients/Sigmoid_3_grad/SigmoidGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_3_grad/SigmoidGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*/
_output_shapes
:���������d*7
_class-
+)loc:@gradients/Sigmoid_3_grad/SigmoidGrad*
T0
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read* 
_output_shapes
::*
N*
T0*
out_type0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:dd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read* 
_output_shapes
::*
N*
T0*
out_type0
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
strides
*
T0*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
data_formatNHWC
�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*/
_output_shapes
:���������d*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*&
_output_shapes
:dd*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*
T0
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
shape: *
_output_shapes
: *
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@Variable*
T0
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
shape: *
_output_shapes
: *
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@Variable*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_1*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_2*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_3*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
�
Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
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
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_4*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
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
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_5*
T0
�
!Variable_6/Adam/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_6*%
valueBdd*    
�
Variable_6/Adam
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
�
Variable_6/Adam/readIdentityVariable_6/Adam*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*&
_output_shapes
:dd*
dtype0*
_class
loc:@Variable_6*%
valueBdd*    
�
Variable_6/Adam_1
VariableV2*
shape:dd*&
_output_shapes
:dd*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*&
_output_shapes
:dd*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*&
_output_shapes
:dd*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_7*
valueBd*    
�
Variable_7/Adam
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
_output_shapes
:d*
dtype0*
_class
loc:@Variable_7*
valueBd*    
�
Variable_7/Adam_1
VariableV2*
shape:d*
_output_shapes
:d*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:d*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_8*
valueB	�*    
�
Variable_8/Adam
VariableV2*
shape:	�*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
z
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
_output_shapes
:	�*
dtype0*
_class
loc:@Variable_8*
valueB	�*    
�
Variable_8/Adam_1
VariableV2*
shape:	�*
_output_shapes
:	�*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_8
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
valueB*    
�
Variable_9/Adam
VariableV2*
shape:*
_output_shapes
:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_9
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
valueB*    
�
Variable_9/Adam_1
VariableV2*
shape:*
_output_shapes
:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_9
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes
:*
_class
loc:@Variable_9*
T0
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
:dd*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_1
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_2
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_3
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_4
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_5
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*&
_output_shapes
:dd*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_6
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_7
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_8
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *
use_locking( *
T0*
_class
loc:@Variable_9
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
use_locking( *
validate_shape(*
T0*
_class
loc:@Variable
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
use_locking( *
validate_shape(*
T0*
_class
loc:@Variable
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*#
_output_shapes
:���������*

DstT0*

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
: *

Tidx0*
T0*
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
accuracy_1*
_output_shapes
: *
N""�
trainable_variables��
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
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0".
	summaries!

cross_entropy:0
accuracy_1:0"
train_op

Adam"�
	variables��
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
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0kݫ�4       ^3\	<Z�>��A*)

cross_entropyJ�?


accuracy_1I�>��JL6       OW��	�%?��A2*)

cross_entropy�,?


accuracy_1�C?PR6       OW��	$sX?��Ad*)

cross_entropy!(?


accuracy_1*?��:7       ���Y	��?��A�*)

cross_entropy��$?


accuracy_1R ?ܥ �7       ���Y	���?��A�*)

cross_entropy=�$?


accuracy_1*?���7       ���Y	�N6@��A�*)

cross_entropywx?


accuracy_1r�*?j��7       ���Y	d�~@��A�*)

cross_entropy��?


accuracy_1!@1?�gD7       ���Y	�c�@��A�*)

cross_entropy�%?


accuracy_1y�5?�J\�7       ���Y	C�A��A�*)

cross_entropy��?


accuracy_1#T4?�Yw�7       ���Y	�XA��A�*)

cross_entropy]h?


accuracy_1Ε4?��K7       ���Y	<D�A��A�*)

cross_entropy�?


accuracy_1�Z5?��,M7       ���Y	���A��A�*)

cross_entropy%�?


accuracy_1$�5?�AG7       ���Y	j(CB��A�*)

cross_entropy��?


accuracy_1xM3?�^7       ���Y	���B��A�*)

cross_entropy�&?


accuracy_1z&7?<L�*7       ���Y	(M�B��A�*)

cross_entropy�?


accuracy_1�n8?/cb7       ���Y	�lC��A�*)

cross_entropyK�?


accuracy_1{�7?H��q7       ���Y	֌dC��A�*)

cross_entropy|?


accuracy_1�39?����7       ���Y	I'�C��A�*)

cross_entropyqx?


accuracy_1&�9?��\�7       ���Y	:��C��A�*)

cross_entropy~�?


accuracy_1%h7?H<�7       ���Y	�"BD��A�*)

cross_entropy}�?


accuracy_1|::?�s�7       ���Y	��D��A�*)

cross_entropy��?


accuracy_1&�9?�?-�7       ���Y	]��D��A�*)

cross_entropy�?


accuracy_1ҽ:? ��X7       ���Y	�y8E��A�*)

cross_entropyk�?


accuracy_1w�2?�%k�7       ���Y	��E��A�*)

cross_entropy��?


accuracy_1|u9?�dR�7       ���Y	�|�E��A�	*)

cross_entropyJ�?


accuracy_1$�6?>4��7       ���Y	B�F��A�	*)

cross_entropyBL?


accuracy_1|::?�^�7       ���Y	��aF��A�
*)

cross_entropy7?


accuracy_1Щ7?�f77       ���Y	P�F��A�
*)

cross_entropy��?


accuracy_1%-8?U�d7       ���Y	���F��A�
*)

cross_entropy�X?


accuracy_1�G<?�e7       ���Y	%�@G��A�*)

cross_entropy��?


accuracy_1(<?ޤj�7       ���Y	���G��A�*)

cross_entropy7�?


accuracy_1҂;?��7       ���Y	"o�G��A�*)

cross_entropy��?


accuracy_1(A;?�/�K7       ���Y	�:H��A�*)

cross_entropy6�?


accuracy_1(A;?{d?�7       ���Y	�+�H��A�*)

cross_entropy�?


accuracy_1}�;?W;P�7       ���Y	G
�H��A�*)

cross_entropy
r?


accuracy_1�G<?��R7       ���Y	ՂI��A�*)

cross_entropy�S?


accuracy_1}�;?m|�7       ���Y	�hI��A�*)

cross_entropy\�?


accuracy_1�G<?�Ee�7       ���Y	�p�I��A�*)

cross_entropy?


accuracy_1)�<?�\�7       ���Y	�I�I��A�*)

cross_entropy]�?


accuracy_1)�<?>���7       ���Y	��HJ��A�*)

cross_entropy~�?


accuracy_1)�=?��J7       ���Y	�;�J��A�*)

cross_entropy�?


accuracy_1(A;?|ʨ�7       ���Y	�:�J��A�*)

cross_entropy�?


accuracy_1~N=?	�g�7       ���Y	�cBK��A�*)

cross_entropy`�?


accuracy_1Ԗ>?�L+L7       ���Y	{�K��A�*)

cross_entropy"�?


accuracy_1(A;?�,�67       ���Y	�(�K��A�*)

cross_entropy],?


accuracy_1�>?��7       ���Y	s�L��A�*)

cross_entropy��
?


accuracy_1Ԗ>?DIK�7       ���Y	:�gL��A�*)

cross_entropy�?


accuracy_1Ԗ>?�	�7       ���Y	/"�L��A�*)

cross_entropy�?


accuracy_1҂;?P��7       ���Y	*��L��A�*)

cross_entropy>�
?


accuracy_1�>?l(7       ���Y	�$@M��A�*)

cross_entropys�
?


accuracy_1)�=?��(�7       ���Y	�b�M��A�*)

cross_entropy]�
?


accuracy_1>?�[I7       ���Y	"��M��A�*)

cross_entropyT�
?


accuracy_1*U>?�c�7       ���Y	&�1N��A�*)

cross_entropy�]
?


accuracy_1*U>?�jw�7       ���Y	K:{N��A�*)

cross_entropy��
?


accuracy_1)�=?�ˊ�7       ���Y	-*�N��A�*)

cross_entropy[�
?


accuracy_1�[??�Sg�7       ���Y	��O��A�*)

cross_entropy��
?


accuracy_1~�<?>�7       ���Y	x�XO��A�*)

cross_entropy�
?


accuracy_1�>?H�-�7       ���Y	}ΡO��A�*)

cross_entropym
?


accuracy_1�[??�J��7       ���Y	��O��A�*)

cross_entropyu�
?


accuracy_1}�;?4�n7       ���Y	�<3P��A�*)

cross_entropy�	?


accuracy_1�[??Db�7       ���Y	mi|P��A�*)

cross_entropy`�
?


accuracy_1)�<?~mѕ7       ���Y	T��P��A�*)

cross_entropy1�	?


accuracy_1*??j�7       ���Y	��$Q��A�*)

cross_entropy�	?


accuracy_1*??��O7       ���Y	��mQ��A�*)

cross_entropy$l	?


accuracy_1*U>?/��7       ���Y	�жQ��A�*)

cross_entropy!�	?


accuracy_1*??֫H{7       ���Y	ܒ�Q��A�*)

cross_entropy<?


accuracy_1(<?_��7       ���Y	��GR��A�*)

cross_entropy�J	?


accuracy_1�>?����7       ���Y	�R��A�*)

cross_entropy�}	?


accuracy_1�[??lO7       ���Y	��R��A�*)

cross_entropy��	?


accuracy_1~N=?�LU�7       ���Y	��"S��A�*)

cross_entropy�]	?


accuracy_1�[??b���7       ���Y	�!kS��A�*)

cross_entropy	?


accuracy_1*??���M7       ���Y	J��S��A�*)

cross_entropy8A	?


accuracy_1�[??�U�7       ���Y	�hT��A�*)

cross_entropy<?


accuracy_1�G<?bT�Q7       ���Y	�'\T��A�*)

cross_entropy��	?


accuracy_1)�=?gDK�7       ���Y	#�T��A�*)

cross_entropy�<	?


accuracy_1*??WA��7       ���Y	ј�T��A�*)

cross_entropy��?


accuracy_1*U>??��Q7       ���Y	H�6U��A�*)

cross_entropyW	?


accuracy_1*??��~�7       ���Y	���U��A�*)

cross_entropy�?


accuracy_1��??��ӓ7       ���Y	�8�U��A�*)

cross_entropy��	?


accuracy_1)�<?J��7       ���Y	(V��A�*)

cross_entropy�!	?


accuracy_1*U>?��n�7       ���Y	��\V��A�*)

cross_entropy��
?


accuracy_1�G<?��c�7       ���Y	he�V��A�*)

cross_entropyt�?


accuracy_1~N=?@��7       ���Y	��V��A� *)

cross_entropy��?


accuracy_1� @?.	�<7       ���Y	�QIW��A� *)

cross_entropy^
?


accuracy_1~�<?,_R�7       ���Y	q|�W��A� *)

cross_entropyij?


accuracy_1+�??���7       ���Y	,�W��A�!*)

cross_entropy�	?


accuracy_1)�=?�`��7       ���Y	�&X��A�!*)

cross_entropy��?


accuracy_1��=?��{7       ���Y	|�oX��A�!*)

cross_entropy�Y?


accuracy_1�[??&���7       ���Y	�X��A�"*)

cross_entropy�G?


accuracy_1*??ȯ"7       ���Y	� Y��A�"*)

cross_entropy:�?


accuracy_1�[??��w7       ���Y	��KY��A�#*)

cross_entropy�
?


accuracy_1~�<?�Ei7       ���Y	�ϪY��A�#*)

cross_entropy�>?


accuracy_1��??)��i7       ���Y	�]�Y��A�#*)

cross_entropy62?


accuracy_1��??��è7       ���Y	��=Z��A�$*)

cross_entropy%�?


accuracy_1)�=?�S
]7       ���Y	��Z��A�$*)

cross_entropy�.?


accuracy_1�>?�b�v7       ���Y	N�Z��A�%*)

cross_entropy2?


accuracy_1�[??�@�f7       ���Y	
�"[��A�%*)

cross_entropyd9	?


accuracy_1)�=?핮7       ���Y	�Nn[��A�%*)

cross_entropy�c?


accuracy_1�[??�%)�7       ���Y	�H�[��A�&*)

cross_entropyg�?


accuracy_1��=?ə�D7       ���Y	�K\��A�&*)

cross_entropy*?


accuracy_1Ԗ>?w��E7       ���Y	`{T\��A�'*)

cross_entropy�?


accuracy_1*??�6w'7       ���Y	Bh�\��A�'*)

cross_entropy=?


accuracy_1��??�u��7       ���Y	�]��A�'*)

cross_entropy|�?


accuracy_1*??���7       ���Y	YQ]��A�(*)

cross_entropy
�?


accuracy_1Ԗ>?�$�B7       ���Y	>�]��A�(*)

cross_entropy��?


accuracy_1Ԗ>?(���7       ���Y	XP�]��A�)*)

cross_entropyg)?


accuracy_1+�??wTݘ7       ���Y	�9^��A�)*)

cross_entropy��?


accuracy_1� @?9N��7       ���Y	�Ά^��A�)*)

cross_entropy	?


accuracy_1� @?or<�7       ���Y	�1�^��A�**)

cross_entropy0�?


accuracy_1>?��f7       ���Y	�� _��A�**)

cross_entropy�@?


accuracy_1+�??���j7       ���Y	4.n_��A�**)

cross_entropy��?


accuracy_1*??nu*�7       ���Y	���_��A�+*)

cross_entropy^a?


accuracy_1+�??k�3;7       ���Y	8�`��A�+*)

cross_entropye�?


accuracy_1Ԗ>?�u�7       ���Y	y�j`��A�,*)

cross_entropyt�?


accuracy_1Ԗ>?j:ޮ7       ���Y	�Ϸ`��A�,*)

cross_entropyG)?


accuracy_1+�??n��O7       ���Y	�a��A�,*)

cross_entropy�	?


accuracy_1�=?��C7       ���Y	
~Ra��A�-*)

cross_entropy�?


accuracy_1*??.�u7       ���Y	��a��A�-*)

cross_entropy��?


accuracy_1*??�^�7       ���Y	���a��A�.*)

cross_entropy2�?


accuracy_1�b@?	�b7       ���Y	B�;b��A�.*)

cross_entropyǾ?


accuracy_1Ԗ>?�=e�7       ���Y	��b��A�.*)

cross_entropy�j?


accuracy_1�b@?!��47       ���Y	�W�b��A�/*)

cross_entropyS8?


accuracy_1*U>?��'7       ���Y	4z0c��A�/*)

cross_entropy��?


accuracy_1��??��7       ���Y	��{c��A�0*)

cross_entropy�r?


accuracy_1>?�;%7       ���Y	R��c��A�0*)

cross_entropy�e?


accuracy_1+�@?弤J7       ���Y	=d��A�0*)

cross_entropy�s?


accuracy_1+�??tߣ�7       ���Y	�bd��A�1*)

cross_entropyD�?


accuracy_1*U>?�x�7       ���Y	�ݭd��A�1*)

cross_entropy��?


accuracy_1Ԗ>?UY�7       ���Y	^<�d��A�2*)

cross_entropy�q?


accuracy_1��=?�g�	7       ���Y	�sBe��A�2*)

cross_entropy΍?


accuracy_1�b@?t�~�7       ���Y	s5�e��A�2*)

cross_entropy�[?


accuracy_1�'A?�/��7       ���Y	�"�e��A�3*)

cross_entropy%Z?


accuracy_1��@?���p7       ���Y	��=f��A�3*)

cross_entropy��?


accuracy_1*??�t7       ���Y	�h�f��A�3*)

cross_entropy��	?


accuracy_1*U>?��R7       ���Y	���f��A�4*)

cross_entropyΆ?


accuracy_1Ԗ>?X7E17       ���Y	��)g��A�4*)

cross_entropy�W?


accuracy_1�>?��G7       ���Y	��vg��A�5*)

cross_entropy��?


accuracy_1+�@?A'#y7       ���Y	,��g��A�5*)

cross_entropy4Q?


accuracy_1�b@?��{7       ���Y	c�h��A�5*)

cross_entropy�;?


accuracy_1��??��@g7       ���Y	V�_h��A�6*)

cross_entropy�j?


accuracy_1�[??MJ�7       ���Y	6�h��A�6*)

cross_entropy!�?


accuracy_1~N=?��c<7       ���Y	� i��A�7*)

cross_entropy��?


accuracy_1)�=?�/7       ���Y	�Zi��A�7*)

cross_entropyz8?


accuracy_1�b@?UX+7       ���Y	Lv�i��A�7*)

cross_entropy�Q?


accuracy_1+�@?�շ�7       ���Y	"q�i��A�8*)

cross_entropy��?


accuracy_1*U>? ��7       ���Y	!vAj��A�8*)

cross_entropy6#?


accuracy_1�'A?��e�7       ���Y	�"�j��A�9*)

cross_entropyZ�?


accuracy_1�>?�0@�7       ���Y	��j��A�9*)

cross_entropy�d?


accuracy_1+�??X�v_7       ���Y	�'k��A�9*)

cross_entropy��?


accuracy_1��=?�c��7       ���Y	��tk��A�:*)

cross_entropy6J?


accuracy_1+�@?�Wg7       ���Y	-�k��A�:*)

cross_entropy�M?


accuracy_1�b@?ҩ�7       ���Y	��l��A�:*)

cross_entropyy?


accuracy_1*U>?�?�7       ���Y	�7ml��A�;*)

cross_entropyQ#?


accuracy_1�b@?Q"I�7       ���Y	z��l��A�;*)

cross_entropyq?


accuracy_1*U>?œ.�7       ���Y	�m��A�<*)

cross_entropy%X?


accuracy_1֪A?�Q)�7       ���Y	�Sm��A�<*)

cross_entropyxB?


accuracy_1*??E��?7       ���Y	X�m��A�<*)

cross_entropy�"?


accuracy_1��@?�<��7       ���Y	��m��A�=*)

cross_entropy�]?


accuracy_1*??�*q7       ���Y	�>n��A�=*)

cross_entropy�	?


accuracy_1��=?E�od7       ���Y	�z�n��A�>*)

cross_entropyg?


accuracy_1+�@?h���