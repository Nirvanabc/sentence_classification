       �K"	  �B���Abrain.Event:2'��      ym�	� �B���A"��
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
dtype0*
_output_shapes
:*%
valueB"����   d      
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
Tshape0*
T0
o
truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d      <   
Z
truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
dtype0*&
_output_shapes
:d<*
seed2 *
T0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:d<*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:d<*
T0
�
Variable
VariableV2*
dtype0*
	container *
shape:d<*&
_output_shapes
:d<*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable
q
Variable/readIdentityVariable*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
R
ConstConst*
dtype0*
_output_shapes
:<*
valueB<*���=
v

Variable_1
VariableV2*
dtype0*
	container *
shape:<*
_output_shapes
:<*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
T0*
validate_shape(*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*/
_output_shapes
:���������d<*
T0
]
addAddConv2DVariable_1/read*/
_output_shapes
:���������d<*
T0
K
ReluReluadd*/
_output_shapes
:���������d<*
T0
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������2<*
ksize
*
T0
q
truncated_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d   <   x   
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
dtype0*&
_output_shapes
:d<x*
seed2 *
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:d<x*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:d<x*
T0
�

Variable_2
VariableV2*
dtype0*
	container *
shape:d<x*&
_output_shapes
:d<x*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
T
Const_1Const*
dtype0*
_output_shapes
:x*
valueBx*���=
v

Variable_3
VariableV2*
dtype0*
	container *
shape:x*
_output_shapes
:x*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
validate_shape(*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*/
_output_shapes
:���������2x*
T0
a
add_1AddConv2D_1Variable_3/read*/
_output_shapes
:���������2x*
T0
O
Relu_1Reluadd_1*/
_output_shapes
:���������2x*
T0
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������x*
ksize
*
T0
q
truncated_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d   x   �   
\
truncated_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
dtype0*'
_output_shapes
:dx�*
seed2 *
T0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*'
_output_shapes
:dx�*
T0
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*'
_output_shapes
:dx�*
T0
�

Variable_4
VariableV2*
dtype0*
	container *
shape:dx�*'
_output_shapes
:dx�*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:dx�*
_class
loc:@Variable_4
x
Variable_4/readIdentity
Variable_4*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
V
Const_2Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_5
VariableV2*
dtype0*
	container *
shape:�*
_output_shapes	
:�*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
Conv2D_2Conv2D	MaxPool_1Variable_4/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*0
_output_shapes
:����������*
T0
b
add_2AddConv2D_2Variable_5/read*0
_output_shapes
:����������*
T0
P
Relu_2Reluadd_2*0
_output_shapes
:����������*
T0
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
paddingSAME*
strides
*0
_output_shapes
:����������*
ksize
*
T0
i
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"`  �  
\
truncated_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
dtype0* 
_output_shapes
:
�0�*
seed2 *
T0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev* 
_output_shapes
:
�0�*
T0
u
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean* 
_output_shapes
:
�0�*
T0
�

Variable_6
VariableV2*
dtype0*
	container *
shape:
�0�* 
_output_shapes
:
�0�*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
�0�*
_class
loc:@Variable_6
q
Variable_6/readIdentity
Variable_6* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
V
Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_7
VariableV2*
dtype0*
	container *
shape:�*
_output_shapes	
:�*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����`  
q
	Reshape_1Reshape	MaxPool_2Reshape_1/shape*(
_output_shapes
:����������0*
Tshape0*
T0
�
MatMulMatMul	Reshape_1Variable_6/read*
transpose_a( *
transpose_b( *(
_output_shapes
:����������*
T0
X
add_3AddMatMulVariable_7/read*(
_output_shapes
:����������*
T0
H
Relu_3Reluadd_3*(
_output_shapes
:����������*
T0
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
S
dropout/ShapeShapeRelu_3*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
dtype0*(
_output_shapes
:����������*
seed2 *
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
dropout/divRealDivRelu_3	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"�     
\
truncated_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*

seed *
dtype0*
_output_shapes
:	�*
seed2 *
T0
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	�*
T0
�

Variable_8
VariableV2*
dtype0*
	container *
shape:	�*
_output_shapes
:	�*
shared_name 
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_8
p
Variable_8/readIdentity
Variable_8*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
T
Const_4Const*
dtype0*
_output_shapes
:*
valueB*���=
v

Variable_9
VariableV2*
dtype0*
	container *
shape:*
_output_shapes
:*
shared_name 
�
Variable_9/AssignAssign
Variable_9Const_4*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
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
MatMul_1MatMuldropout/mulVariable_8/read*
transpose_a( *
transpose_b( *'
_output_shapes
:���������*
T0
Y
add_4AddMatMul_1Variable_9/read*'
_output_shapes
:���������*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
J
ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
L
Shape_1Shapeadd_4*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*

axis *
N*
_output_shapes
:*
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
q
concatConcatV2concat/values_0Sliceconcat/axis*

Tidx0*
_output_shapes
:*
N*
T0
l
	Reshape_2Reshapeadd_4concat*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*

axis *
N*
_output_shapes
:*
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
O
concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
_output_shapes
:*
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
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*

axis *
N*
_output_shapes
:*
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
Const_5Const*
dtype0*
_output_shapes
:*
valueB: 
^
MeanMean	Reshape_4Const_5*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
dtype0*
_output_shapes
: *
valueB Bcross_entropy
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
�
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_4*
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
gradients/add_4_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_4_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_4_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
T0
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
transpose_b(*(
_output_shapes
:����������*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_4_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes
:	�*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:����������*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
f
 gradients/dropout/div_grad/ShapeShapeRelu_3*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
gradients/dropout/div_grad/NegNegRelu_3*(
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
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
�
gradients/Relu_3_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:����������*
T0
`
gradients/add_3_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_3_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes	
:�*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*(
_output_shapes
:����������0*
T0
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( * 
_output_shapes
:
�0�*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps* 
_output_shapes
:
�0�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_2*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_2 gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
paddingSAME*
strides
*0
_output_shapes
:����������*
ksize
*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:����������*
T0
b
gradients/add_2_grad/ShapeShapeConv2D_2*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*0
_output_shapes
:����������*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes	
:�*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeN	MaxPool_1Variable_4/read*
out_type0*
N* 
_output_shapes
::*
T0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1 gradients/Conv2D_2_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������x*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*'
_output_shapes
:dx�*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/Conv2D_2_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������2x*
ksize
*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:���������2x*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:x
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*/
_output_shapes
:���������2x*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:x*
Tshape0*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_output_shapes
:���������2x*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
:x*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
N* 
_output_shapes
::*
T0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������2<*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:d<x*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������d<*
ksize
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:���������d<*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:<
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:���������d<*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:<*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*/
_output_shapes
:���������d<*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:<*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N* 
_output_shapes
::*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:d<*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
{
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable
�
beta1_power
VariableV2*
shared_name *
shape: *
_class
loc:@Variable*
dtype0*
	container *
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
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@Variable*
T0
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@Variable
�
beta2_power
VariableV2*
shared_name *
shape: *
_class
loc:@Variable*
dtype0*
	container *
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
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Variable/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<*%
valueBd<*    *
_class
loc:@Variable
�
Variable/Adam
VariableV2*
shared_name *
shape:d<*
_class
loc:@Variable*
dtype0*
	container *&
_output_shapes
:d<
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<*%
valueBd<*    *
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
shared_name *
shape:d<*
_class
loc:@Variable*
dtype0*
	container *&
_output_shapes
:d<
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:<*
valueB<*    *
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
shared_name *
shape:<*
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes
:<
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:<*
valueB<*    *
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
shared_name *
shape:<*
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes
:<
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<x*%
valueBd<x*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
shared_name *
shape:d<x*
_class
loc:@Variable_2*
dtype0*
	container *&
_output_shapes
:d<x
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<x*%
valueBd<x*    *
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
shared_name *
shape:d<x*
_class
loc:@Variable_2*
dtype0*
	container *&
_output_shapes
:d<x
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:x*
valueBx*    *
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
shared_name *
shape:x*
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes
:x
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:x*
valueBx*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
shared_name *
shape:x*
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes
:x
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*'
_output_shapes
:dx�*&
valueBdx�*    *
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
shared_name *
shape:dx�*
_class
loc:@Variable_4*
dtype0*
	container *'
_output_shapes
:dx�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:dx�*
_class
loc:@Variable_4
�
Variable_4/Adam/readIdentityVariable_4/Adam*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*'
_output_shapes
:dx�*&
valueBdx�*    *
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
shared_name *
shape:dx�*
_class
loc:@Variable_4*
dtype0*
	container *'
_output_shapes
:dx�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:dx�*
_class
loc:@Variable_4
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes	
:�
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_5
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Adam_1
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes	
:�
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_5
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
!Variable_6/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
�0�*
valueB
�0�*    *
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
shared_name *
shape:
�0�*
_class
loc:@Variable_6*
dtype0*
	container * 
_output_shapes
:
�0�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
�0�*
_class
loc:@Variable_6
{
Variable_6/Adam/readIdentityVariable_6/Adam* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
�0�*
valueB
�0�*    *
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
shared_name *
shape:
�0�*
_class
loc:@Variable_6*
dtype0*
	container * 
_output_shapes
:
�0�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
�0�*
_class
loc:@Variable_6

Variable_6/Adam_1/readIdentityVariable_6/Adam_1* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_7*
dtype0*
	container *
_output_shapes	
:�
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7
v
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam_1
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_7*
dtype0*
	container *
_output_shapes	
:�
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_8
�
Variable_8/Adam
VariableV2*
shared_name *
shape:	�*
_class
loc:@Variable_8*
dtype0*
	container *
_output_shapes
:	�
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_8
z
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_8
�
Variable_8/Adam_1
VariableV2*
shared_name *
shape:	�*
_class
loc:@Variable_8*
dtype0*
	container *
_output_shapes
:	�
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_8
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_9
�
Variable_9/Adam
VariableV2*
shared_name *
shape:*
_class
loc:@Variable_9*
dtype0*
	container *
_output_shapes
:
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
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
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_9
�
Variable_9/Adam_1
VariableV2*
shared_name *
shape:*
_class
loc:@Variable_9*
dtype0*
	container *
_output_shapes
:
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
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
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *��8
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:d<*
use_locking( *
_class
loc:@Variable*
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:<*
use_locking( *
_class
loc:@Variable_1*
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:d<x*
use_locking( *
_class
loc:@Variable_2*
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:x*
use_locking( *
_class
loc:@Variable_3*
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *'
_output_shapes
:dx�*
use_locking( *
_class
loc:@Variable_4*
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
_class
loc:@Variable_5*
T0
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
�0�*
use_locking( *
_class
loc:@Variable_6*
T0
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
_class
loc:@Variable_7*
T0
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
_class
loc:@Variable_8*
T0
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
_class
loc:@Variable_9*
T0
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
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
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
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
v
ArgMaxArgMaxadd_4ArgMax/dimension*

Tidx0*#
_output_shapes
:���������*
T0*
output_type0	
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*#
_output_shapes
:���������*
T0*
output_type0	
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
Const_6Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_6*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
dtype0*
_output_shapes
: *
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
N"��LNq     `E�	��C���AJ�
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
dtype0*
_output_shapes
:*%
valueB"����   d      
l
ReshapeReshapexReshape/shape*/
_output_shapes
:���������d*
Tshape0*
T0
o
truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d      <   
Z
truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
dtype0*&
_output_shapes
:d<*
seed2 *
T0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:d<*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:d<*
T0
�
Variable
VariableV2*
dtype0*
	container *
shape:d<*&
_output_shapes
:d<*
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:d<*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0
q
Variable/readIdentityVariable*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
R
ConstConst*
dtype0*
_output_shapes
:<*
valueB<*���=
v

Variable_1
VariableV2*
dtype0*
	container *
shape:<*
_output_shapes
:<*
shared_name 
�
Variable_1/AssignAssign
Variable_1Const*
_output_shapes
:<*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*/
_output_shapes
:���������d<*
T0
]
addAddConv2DVariable_1/read*/
_output_shapes
:���������d<*
T0
K
ReluReluadd*/
_output_shapes
:���������d<*
T0
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������2<*
ksize
*
T0
q
truncated_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d   <   x   
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
dtype0*&
_output_shapes
:d<x*
seed2 *
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:d<x*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:d<x*
T0
�

Variable_2
VariableV2*
dtype0*
	container *
shape:d<x*&
_output_shapes
:d<x*
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*&
_output_shapes
:d<x*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0
w
Variable_2/readIdentity
Variable_2*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
T
Const_1Const*
dtype0*
_output_shapes
:x*
valueBx*���=
v

Variable_3
VariableV2*
dtype0*
	container *
shape:x*
_output_shapes
:x*
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
_output_shapes
:x*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DMaxPoolVariable_2/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*/
_output_shapes
:���������2x*
T0
a
add_1AddConv2D_1Variable_3/read*/
_output_shapes
:���������2x*
T0
O
Relu_1Reluadd_1*/
_output_shapes
:���������2x*
T0
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������x*
ksize
*
T0
q
truncated_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d   x   �   
\
truncated_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
dtype0*'
_output_shapes
:dx�*
seed2 *
T0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*'
_output_shapes
:dx�*
T0
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*'
_output_shapes
:dx�*
T0
�

Variable_4
VariableV2*
dtype0*
	container *
shape:dx�*'
_output_shapes
:dx�*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*'
_output_shapes
:dx�*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0
x
Variable_4/readIdentity
Variable_4*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
V
Const_2Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_5
VariableV2*
dtype0*
	container *
shape:�*
_output_shapes	
:�*
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
Conv2D_2Conv2D	MaxPool_1Variable_4/read*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*0
_output_shapes
:����������*
T0
b
add_2AddConv2D_2Variable_5/read*0
_output_shapes
:����������*
T0
P
Relu_2Reluadd_2*0
_output_shapes
:����������*
T0
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
paddingSAME*
strides
*0
_output_shapes
:����������*
ksize
*
T0
i
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"`  �  
\
truncated_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
dtype0* 
_output_shapes
:
�0�*
seed2 *
T0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev* 
_output_shapes
:
�0�*
T0
u
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean* 
_output_shapes
:
�0�*
T0
�

Variable_6
VariableV2*
dtype0*
	container *
shape:
�0�* 
_output_shapes
:
�0�*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3* 
_output_shapes
:
�0�*
validate_shape(*
use_locking(*
_class
loc:@Variable_6*
T0
q
Variable_6/readIdentity
Variable_6* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
V
Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_7
VariableV2*
dtype0*
	container *
shape:�*
_output_shapes	
:�*
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_7*
T0
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����`  
q
	Reshape_1Reshape	MaxPool_2Reshape_1/shape*(
_output_shapes
:����������0*
Tshape0*
T0
�
MatMulMatMul	Reshape_1Variable_6/read*
transpose_a( *
transpose_b( *(
_output_shapes
:����������*
T0
X
add_3AddMatMulVariable_7/read*(
_output_shapes
:����������*
T0
H
Relu_3Reluadd_3*(
_output_shapes
:����������*
T0
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
S
dropout/ShapeShapeRelu_3*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
_
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
dtype0*(
_output_shapes
:����������*
seed2 *
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
dropout/divRealDivRelu_3	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:����������*
T0
i
truncated_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"�     
\
truncated_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*

seed *
dtype0*
_output_shapes
:	�*
seed2 *
T0
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	�*
T0
�

Variable_8
VariableV2*
dtype0*
	container *
shape:	�*
_output_shapes
:	�*
shared_name 
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8*
T0
p
Variable_8/readIdentity
Variable_8*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
T
Const_4Const*
dtype0*
_output_shapes
:*
valueB*���=
v

Variable_9
VariableV2*
dtype0*
	container *
shape:*
_output_shapes
:*
shared_name 
�
Variable_9/AssignAssign
Variable_9Const_4*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9*
T0
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
MatMul_1MatMuldropout/mulVariable_8/read*
transpose_a( *
transpose_b( *'
_output_shapes
:���������*
T0
Y
add_4AddMatMul_1Variable_9/read*'
_output_shapes
:���������*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
J
ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
L
Shape_1Shapeadd_4*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*

axis *
N*
_output_shapes
:*
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
Index0*
T0
b
concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
q
concatConcatV2concat/values_0Sliceconcat/axis*

Tidx0*
_output_shapes
:*
N*
T0
l
	Reshape_2Reshapeadd_4concat*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*

axis *
N*
_output_shapes
:*
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
Index0*
T0
d
concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
O
concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
_output_shapes
:*
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
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*

axis *
N*
_output_shapes
:*
T0
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
Const_5Const*
dtype0*
_output_shapes
:*
valueB: 
^
MeanMean	Reshape_4Const_5*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
`
cross_entropy/tagsConst*
dtype0*
_output_shapes
: *
valueB Bcross_entropy
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
�
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_4*
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
gradients/add_4_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_4_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_4_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
T0
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
transpose_b(*(
_output_shapes
:����������*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_4_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes
:	�*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:����������*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	�*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
f
 gradients/dropout/div_grad/ShapeShapeRelu_3*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
gradients/dropout/div_grad/NegNegRelu_3*(
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
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
�
gradients/Relu_3_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:����������*
T0
`
gradients/add_3_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_3_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes	
:�*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*(
_output_shapes
:����������0*
T0
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( * 
_output_shapes
:
�0�*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps* 
_output_shapes
:
�0�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_2*
out_type0*
_output_shapes
:*
T0
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_2 gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
paddingSAME*
strides
*0
_output_shapes
:����������*
ksize
*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:����������*
T0
b
gradients/add_2_grad/ShapeShapeConv2D_2*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*0
_output_shapes
:����������*
Tshape0*
T0
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*0
_output_shapes
:����������*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes	
:�*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeN	MaxPool_1Variable_4/read*
out_type0*
N* 
_output_shapes
::*
T0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1 gradients/Conv2D_2_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:���������x*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*'
_output_shapes
:dx�*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/Conv2D_2_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������2x*
ksize
*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:���������2x*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:x
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*/
_output_shapes
:���������2x*
Tshape0*
T0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:x*
Tshape0*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_output_shapes
:���������2x*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
:x*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
N* 
_output_shapes
::*
T0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:���������2<*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:d<x*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
strides
*/
_output_shapes
:���������d<*
ksize
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:���������d<*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:<
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:���������d<*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:<*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*/
_output_shapes
:���������d<*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:<*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N* 
_output_shapes
::*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*J
_output_shapes8
6:4������������������������������������*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������d*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:d<*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
{
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable
�
beta1_power
VariableV2*
shared_name *
shape: *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@Variable*
T0
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@Variable
�
beta2_power
VariableV2*
shared_name *
shape: *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Variable/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<*%
valueBd<*    *
_class
loc:@Variable
�
Variable/Adam
VariableV2*
shared_name *
shape:d<*
_class
loc:@Variable*
dtype0*
	container *&
_output_shapes
:d<
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*&
_output_shapes
:d<*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0
{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<*%
valueBd<*    *
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
shared_name *
shape:d<*
_class
loc:@Variable*
dtype0*
	container *&
_output_shapes
:d<
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*&
_output_shapes
:d<*
validate_shape(*
use_locking(*
_class
loc:@Variable*
T0

Variable/Adam_1/readIdentityVariable/Adam_1*&
_output_shapes
:d<*
_class
loc:@Variable*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:<*
valueB<*    *
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
shared_name *
shape:<*
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes
:<
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes
:<*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:<*
valueB<*    *
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
shared_name *
shape:<*
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes
:<
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_output_shapes
:<*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
T0
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:<*
_class
loc:@Variable_1*
T0
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<x*%
valueBd<x*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
shared_name *
shape:d<x*
_class
loc:@Variable_2*
dtype0*
	container *&
_output_shapes
:d<x
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*&
_output_shapes
:d<x*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:d<x*%
valueBd<x*    *
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
shared_name *
shape:d<x*
_class
loc:@Variable_2*
dtype0*
	container *&
_output_shapes
:d<x
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*&
_output_shapes
:d<x*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:x*
valueBx*    *
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
shared_name *
shape:x*
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes
:x
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes
:x*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:x*
valueBx*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
shared_name *
shape:x*
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes
:x
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_output_shapes
:x*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
T0
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:x*
_class
loc:@Variable_3*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*'
_output_shapes
:dx�*&
valueBdx�*    *
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
shared_name *
shape:dx�*
_class
loc:@Variable_4*
dtype0*
	container *'
_output_shapes
:dx�
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*'
_output_shapes
:dx�*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam/readIdentityVariable_4/Adam*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*'
_output_shapes
:dx�*&
valueBdx�*    *
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
shared_name *
shape:dx�*
_class
loc:@Variable_4*
dtype0*
	container *'
_output_shapes
:dx�
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*'
_output_shapes
:dx�*
validate_shape(*
use_locking(*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*'
_output_shapes
:dx�*
_class
loc:@Variable_4*
T0
�
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Adam
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes	
:�
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_5
�
Variable_5/Adam_1
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes	
:�
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_5*
T0
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
!Variable_6/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
�0�*
valueB
�0�*    *
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
shared_name *
shape:
�0�*
_class
loc:@Variable_6*
dtype0*
	container * 
_output_shapes
:
�0�
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros* 
_output_shapes
:
�0�*
validate_shape(*
use_locking(*
_class
loc:@Variable_6*
T0
{
Variable_6/Adam/readIdentityVariable_6/Adam* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
�0�*
valueB
�0�*    *
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
shared_name *
shape:
�0�*
_class
loc:@Variable_6*
dtype0*
	container * 
_output_shapes
:
�0�
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros* 
_output_shapes
:
�0�*
validate_shape(*
use_locking(*
_class
loc:@Variable_6*
T0

Variable_6/Adam_1/readIdentityVariable_6/Adam_1* 
_output_shapes
:
�0�*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_7*
dtype0*
	container *
_output_shapes	
:�
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_7*
T0
v
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam_1
VariableV2*
shared_name *
shape:�*
_class
loc:@Variable_7*
dtype0*
	container *
_output_shapes	
:�
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes	
:�*
validate_shape(*
use_locking(*
_class
loc:@Variable_7*
T0
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_8
�
Variable_8/Adam
VariableV2*
shared_name *
shape:	�*
_class
loc:@Variable_8*
dtype0*
	container *
_output_shapes
:	�
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8*
T0
z
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_8
�
Variable_8/Adam_1
VariableV2*
shared_name *
shape:	�*
_class
loc:@Variable_8*
dtype0*
	container *
_output_shapes
:	�
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
_output_shapes
:	�*
validate_shape(*
use_locking(*
_class
loc:@Variable_8*
T0
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_9
�
Variable_9/Adam
VariableV2*
shared_name *
shape:*
_class
loc:@Variable_9*
dtype0*
	container *
_output_shapes
:
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9*
T0
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_9
�
Variable_9/Adam_1
VariableV2*
shared_name *
shape:*
_class
loc:@Variable_9*
dtype0*
	container *
_output_shapes
:
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@Variable_9*
T0
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes
:*
_class
loc:@Variable_9*
T0
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *��8
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:d<*
use_locking( *
_class
loc:@Variable*
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:<*
use_locking( *
_class
loc:@Variable_1*
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:d<x*
use_locking( *
_class
loc:@Variable_2*
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:x*
use_locking( *
_class
loc:@Variable_3*
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *'
_output_shapes
:dx�*
use_locking( *
_class
loc:@Variable_4*
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
_class
loc:@Variable_5*
T0
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
�0�*
use_locking( *
_class
loc:@Variable_6*
T0
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
_class
loc:@Variable_7*
T0
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
_class
loc:@Variable_8*
T0
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
_class
loc:@Variable_9*
T0
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
: *
validate_shape(*
use_locking( *
_class
loc:@Variable*
T0
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
: *
validate_shape(*
use_locking( *
_class
loc:@Variable*
T0
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
v
ArgMaxArgMaxadd_4ArgMax/dimension*

Tidx0*#
_output_shapes
:���������*
output_type0	*
T0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
Cast_1CastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

Q
Const_6Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_6*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
Z
accuracy_1/tagsConst*
dtype0*
_output_shapes
: *
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
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0���4       ^3\	(;�P���A*)

cross_entropyeжD


accuracy_1{�>��� 6       OW��	�ޭ����A*)

cross_entropy:YcD


accuracy_1��>���6       OW��	r ����A(*)

cross_entropy�%XC


accuracy_1�G?,��)6       OW��	 =���A<*)

cross_entropy5B�C


accuracy_1�|�>��w�6       OW��	��)���AP*)

cross_entropyV�B


accuracy_1-?k��6       OW��	�f`���Ad*)

cross_entropy;�B


accuracy_1��?̉6       OW��	%�����Ax*)

cross_entropy�2C


accuracy_1�t?!ܝ7       ���Y	���̯��A�*)

cross_entropy�ѭB


accuracy_1�?�dD�7       ���Y	�	g���A�*)

cross_entropy�݅B


accuracy_1��?�R�7       ���Y	�.�9���A�*)

cross_entropy��kB


accuracy_1X?�~l�7       ���Y	=�Up���A�*)

cross_entropy��B


accuracy_1=
?k)m�7       ���Y	��ʦ���A�*)

cross_entropy���B


accuracy_133?��_�7       ���Y	]�Iݰ��A�*)

cross_entropy��B


accuracy_1�t?W�I�7       ���Y	�b���A�*)

cross_entropy���B


accuracy_1=
?�?H�7       ���Y	�S�J���A�*)

cross_entropy�B


accuracy_1��?����7       ���Y	�t����A�*)

cross_entropyO��B


accuracy_1}??�_�7       ���Y	Y�����A�*)

cross_entropy�Z�B


accuracy_1/?��7       ���Y	�Z�A�*)

cross_entropy�z�B


accuracy_1+�?<C�7       ���Y	�'�$���A�*)

cross_entropy�WB


accuracy_1P�?��7       ���Y	ICS[���A�*)

cross_entropy:��A


accuracy_1+�?�y7       ���Y	w�ᑲ��A�*)

cross_entropy���A


accuracy_1�?^P��7       ���Y	�Ȳ��A�*)

cross_entropy�c�A


accuracy_1��?��c�7       ���Y	
����A�*)

cross_entropyt!�A


accuracy_1j�?}X��7       ���Y	�E�5���A�*)

cross_entropy\_�A


accuracy_15^?)�)7       ���Y	��0l���A�*)

cross_entropy��A


accuracy_1u�?(�p�7       ���Y	#�ޢ���A�*)

cross_entropyَEB


accuracy_1�&?�>��7       ���Y	k��ٳ��A�*)

cross_entropy�B


accuracy_1�E?��D�7       ���Y	^H���A�*)

cross_entropy���A


accuracy_1��?�,7       ���Y	�\�F���A�*)

cross_entropy�T�A


accuracy_1�t?�vP7       ���Y	<}���A�*)

cross_entropy%��A


accuracy_1�E?�Z]�7       ���Y	��̳���A�*)

cross_entropy&��A


accuracy_1X9?����7       ���Y	@hR괐�A�*)

cross_entropy9��A


accuracy_1=
?�9��7       ���Y	�c� ���A�*)

cross_entropyŭ�A


accuracy_1j�?똍7       ���Y	��xW���A�*)

cross_entropy���A


accuracy_1��?h��{7       ���Y	#�����A�*)

cross_entropyjz�A


accuracy_1��?Wb$�7       ���Y	6�ĵ��A�*)

cross_entropy­�A


accuracy_1�?�q-7       ���Y	#������A�*)

cross_entropy�؛A


accuracy_1�C?S^t7       ���Y	Sr�1���A�*)

cross_entropy $�A


accuracy_1q=
?�k�7       ���Y	��Yh���A�*)

cross_entropy��A


accuracy_1Nb?���7       ���Y	=�����A�*)

cross_entropys��A


accuracy_1Nb?2�777       ���Y	��;ն��A�*)

cross_entropyd�`A


accuracy_1sh?`+O{7       ���Y	�o����A�*)

cross_entropy��cA


accuracy_1X9?T�A�7       ���Y	��B���A�*)

cross_entropy��A


accuracy_1��?�
�g7       ���Y	��=y���A�*)

cross_entropy3�hA


accuracy_1)\?k'�"7       ���Y	Cnʯ���A�*)

cross_entropy�^FA


accuracy_1��?��g^7       ���Y	 ��淐�A�*)

cross_entropyy�4A


accuracy_1��?�h�Y7       ���Y	�*_���A�*)

cross_entropy�0A


accuracy_1�C?�:D7       ���Y	�*T���A�*)

cross_entropy �$A


accuracy_1� ?�z�7       ���Y	
֊���A�*)

cross_entropy�aA


accuracy_1D�?��$�7       ���Y	��o����A�*)

cross_entropy�A


accuracy_1V?����7       ���Y	{�����A�*)

cross_entropy�=A


accuracy_1��?u�7       ���Y	41/���A�*)

cross_entropya$)A


accuracy_1�I?���}7       ���Y	���e���A�*)

cross_entropy:s!A


accuracy_1�I?;0H�7       ���Y	𥂜���A�*)

cross_entropy��A


accuracy_1�?:B^7       ���Y	�$ӹ��A�*)

cross_entropy#A


accuracy_1�r?Ӌ��7       ���Y	q�	���A�*)

cross_entropy�A


accuracy_11?xƵc7       ���Y	�A�@���A�*)

cross_entropy�V�@


accuracy_1�?��+&7       ���Y	��yw���A�*)

cross_entropy7A


accuracy_1�I?�e�7       ���Y	������A�	*)

cross_entropy��A


accuracy_1D�?�;�7       ���Y	��亐�A�	*)

cross_entropy�A


accuracy_1��
?NP��7       ���Y	#K����A�	*)

cross_entropy��A


accuracy_11?ys�7       ���Y	��R���A�	*)

cross_entropy���@


accuracy_1h�?�5�b7       ���Y	c�)����A�	*)

cross_entropy�O�@


accuracy_1�?���n7       ���Y	Y�ݿ���A�	*)

cross_entropy��@


accuracy_1q=
?&#��7       ���Y	Qe�����A�
*)

cross_entropy��@


accuracy_1q=
?o�C�7       ���Y	��H-���A�
*)

cross_entropy�ٿ@


accuracy_1��
?بi7       ���Y	nmd���A�
*)

cross_entropy6�@


accuracy_1?D��7       ���Y	�>욼��A�
*)

cross_entropy3�@


accuracy_1��?��7       ���Y	��Ѽ��A�
*)

cross_entropy�@


accuracy_1?�h$7       ���Y	ʔ[���A�
*)

cross_entropy�3�@


accuracy_1�I?k�x�7       ���Y	��>���A�
*)

cross_entropy�S�@


accuracy_1h�?g��7       ���Y	���u���A�*)

cross_entropy���@


accuracy_1��
?=��z7       ���Y	�Yd����A�*)

cross_entropyR8�@


accuracy_1   ?�Hۮ7       ���Y	��㽐�A�*)

cross_entropy�R�@


accuracy_1'1?�ǲ7       ���Y	}B����A�*)

cross_entropyu�@


accuracy_1�C?Nr#o7       ���Y	AeP���A�*)

cross_entropy��@


accuracy_1�?�b'�7       ���Y	U�����A�*)

cross_entropy�"y@


accuracy_1�?�!�>7       ���Y	6�ν���A�*)

cross_entropyT�x@


accuracy_1��
?�˚�7       ���Y	.�z����A�*)

cross_entropy���@


accuracy_1�O?S{m�7       ���Y	�3;+���A�*)

cross_entropy�x@


accuracy_1ff?�M�7       ���Y	N�a���A�*)

cross_entropyqb@


accuracy_1��
?�Q�7       ���Y	�������A�*)

cross_entropyT�k@


accuracy_17�?g�?s7       ���Y	V΁Ͽ��A�*)

cross_entropy���@


accuracy_1�M?btQ$7       ���Y	J�7���A�*)

cross_entropy[8�@


accuracy_1q=
?��л7       ���Y	��<���A�*)

cross_entropy�@


accuracy_1�?��f7       ���Y	��os���A�*)

cross_entropyLhm@


accuracy_1V?�_�7       ���Y	Qk����A�*)

cross_entropy�{�@


accuracy_1V?j'Ӯ7       ���Y	"�����A�*)

cross_entropy���@


accuracy_1�O?1��<7       ���Y	T�:���A�*)

cross_entropy3�@


accuracy_1V?�̈�7       ���Y	0F�M���A�*)

cross_entropy�U�@


accuracy_1�~
?����7       ���Y	&������A�*)

cross_entropyB��A


accuracy_1o?>[�G7       ���Y	��Q����A�*)

cross_entropy̎�B


accuracy_1�?�K�7       ���Y	������A�*)

cross_entropy�.�A


accuracy_1%?�JX7       ���Y	�c(�A�*)

cross_entropy%��@


accuracy_1��
?�fS�7       ���Y	�|�^�A�*)

cross_entropy�>A


accuracy_1�x	?��%7       ���Y	��9��A�*)

cross_entropy��A


accuracy_1L7	?L���7       ���Y	���A�*)

cross_entropy��A


accuracy_1ˡ?�47�7       ���Y	���Ð�A�*)

cross_entropyTA


accuracy_1�l?Ƿ@7       ���Y	r�8Ð�A�*)

cross_entropy*k�@


accuracy_1y�? }@7       ���Y	vi�oÐ�A�*)

cross_entropyz��@


accuracy_1�?P��;7       ���Y	#'K�Ð�A�*)

cross_entropy9��@


accuracy_1�l?B�|7       ���Y	�+�Ð�A�*)

cross_entropy�'�@


accuracy_1��
?�I�7       ���Y	���Đ�A�*)

cross_entropy���@


accuracy_1ˡ?#7�/7       ���Y	���JĐ�A�*)

cross_entropyM	A


accuracy_1��
?ĲET7       ���Y	o�X�Đ�A�*)

cross_entropyN�A


accuracy_11?���d7       ���Y	Zs�Đ�A�*)

cross_entropyylA


accuracy_1T�?��H7       ���Y	$T��Đ�A�*)

cross_entropy��A


accuracy_1ˡ?A'��7       ���Y	�%Ő�A�*)

cross_entropy'0�@


accuracy_1'1?]�87       ���Y	�@�\Ő�A�*)

cross_entropy�@


accuracy_1y�?�UB�7       ���Y	���Ő�A�*)

cross_entropy(��@


accuracy_1
�?��27       ���Y	P���Ő�A�*)

cross_entropy���@


accuracy_1�?��z7       ���Y	H9� Ɛ�A�*)

cross_entropy��@


accuracy_1�~
?�47       ���Y	BTk7Ɛ�A�*)

cross_entropyj��@


accuracy_1�x	?K��7       ���Y	��HnƐ�A�*)

cross_entropy��@


accuracy_1��	?=�7       ���Y	���Ɛ�A�*)

cross_entropy���@


accuracy_1�C?��7       ���Y	����Ɛ�A�*)

cross_entropy٭8@


accuracy_1h�?��7       ���Y	�vǐ�A�*)

cross_entropy�4@


accuracy_1{?Rɇ�7       ���Y	!bIǐ�A�*)

cross_entropy cS@


accuracy_1h�?koE'7       ���Y	om�ǐ�A�*)

cross_entropy�ߒ@


accuracy_1�O??>�q7       ���Y	Ҝ��ǐ�A�*)

cross_entropy���@


accuracy_1�~
?I�S�7       ���Y	d�c�ǐ�A�*)

cross_entropy��@


accuracy_1�?�� 17       ���Y	��N$Ȑ�A�*)

cross_entropy��@


accuracy_1'1?Km�7       ���Y	ؿ�ZȐ�A�*)

cross_entropy^��@


accuracy_1q=
?��&�7       ���Y	Ab��Ȑ�A�*)

cross_entropyϷ@


accuracy_1L7	?Y�C7       ���Y	E�Ȑ�A�*)

cross_entropy��@


accuracy_19�?n�77       ���Y	W���Ȑ�A�*)

cross_entropy@


accuracy_1�O?sߒ�7       ���Y	��5ɐ�A�*)

cross_entropy���@


accuracy_1Nb?f���7       ���Y	��:lɐ�A�*)

cross_entropy�*�@


accuracy_11?ɴ'�7       ���Y	�DѢɐ�A�*)

cross_entropy���@


accuracy_1�?_|17       ���Y	a���ɐ�A�*)

cross_entropy'��@


accuracy_1{?V��,7       ���Y	$�(ʐ�A�*)

cross_entropy5��@


accuracy_11?w�c!7       ���Y	*��Fʐ�A�*)

cross_entropy�~�@


accuracy_1h�? ɞ7       ���Y	�z}ʐ�A�*)

cross_entropy�2�@


accuracy_1�?����7       ���Y	{(�ʐ�A�*)

cross_entropy9��@


accuracy_1��	?z'�r7       ���Y	����ʐ�A�*)

cross_entropy��@


accuracy_1��?K�w�7       ���Y	r7!ː�A�*)

cross_entropy�A�@


accuracy_1��?R`7       ���Y	oXː�A�*)

cross_entropy��@


accuracy_1'1?E٠7       ���Y	��ώː�A�*)

cross_entropy�߮@


accuracy_1� ?��w�7       ���Y	��ː�A�*)

cross_entropy��@


accuracy_1�&?&4!7       ���Y	~�)�ː�A�*)

cross_entropy�@


accuracy_1Nb?q�B�7       ���Y	��2̐�A�*)

cross_entropy�J�@


accuracy_1h�?�$��7       ���Y	�-i̐�A�*)

cross_entropyw�@


accuracy_1�?㼠#7       ���Y	�V��̐�A�*)

cross_entropy�[�@


accuracy_1q=
?fN�7       ���Y	�7#�̐�A�*)

cross_entropy���@


accuracy_1��?��J7       ���Y	��͐�A�*)

cross_entropyNͫ@


accuracy_1^�	?b���7       ���Y	�rC͐�A�*)

cross_entropy-N�@


accuracy_1�?�?37       ���Y	O�y͐�A�*)

cross_entropyh��@


accuracy_1!�?]4�7       ���Y	�g�͐�A�*)

cross_entropy��]@


accuracy_1-?X� �7       ���Y	�[�͐�A�*)

cross_entropy9?R@


accuracy_1� ?�]ɑ7       ���Y	�c�ΐ�A�*)

cross_entropy�v@


accuracy_1� ?4�(