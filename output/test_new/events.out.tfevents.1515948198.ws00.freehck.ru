       �K"	  �)��Abrain.Event:2��{�x     ��,	b��)��A"�
n
xPlaceholder*
dtype0*,
_output_shapes
:����������*!
shape:����������
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����   ,     
m
ReshapeReshapexReshape/shape*0
_output_shapes
:����������*
Tshape0*
T0
o
truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *

seed *(
_output_shapes
:��*
T0
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*(
_output_shapes
:��*
T0
w
truncated_normalAddtruncated_normal/multruncated_normal/mean*(
_output_shapes
:��*
T0
�
Variable
VariableV2*
dtype0*(
_output_shapes
:��*
	container *
shared_name *
shape:��
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
s
Variable/readIdentityVariable*(
_output_shapes
:��*
_class
loc:@Variable*
T0
T
ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shared_name *
shape:�
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
l
Variable_1/readIdentity
Variable_1*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
}
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
P
ReluReluBiasAdd*0
_output_shapes
:����������*
T0
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *

seed *(
_output_shapes
:��*
T0
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*(
_output_shapes
:��*
T0
�

Variable_2
VariableV2*
dtype0*(
_output_shapes
:��*
	container *
shared_name *
shape:��
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
y
Variable_2/readIdentity
Variable_2*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
V
Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_3
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shared_name *
shape:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
l
Variable_3/readIdentity
Variable_3*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_1Relu	BiasAdd_1*0
_output_shapes
:����������*
T0
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *

seed *(
_output_shapes
:��*
T0
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*(
_output_shapes
:��*
T0
�

Variable_4
VariableV2*
dtype0*(
_output_shapes
:��*
	container *
shared_name *
shape:��
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
y
Variable_4/readIdentity
Variable_4*(
_output_shapes
:��*
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
dtype0*
_output_shapes	
:�*
	container *
shared_name *
shape:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_2Relu	BiasAdd_2*0
_output_shapes
:����������*
T0
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *

seed *(
_output_shapes
:��*
T0
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*(
_output_shapes
:��*
T0
�

Variable_6
VariableV2*
dtype0*(
_output_shapes
:��*
	container *
shared_name *
shape:��
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
y
Variable_6/readIdentity
Variable_6*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
V
Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_7
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shared_name *
shape:�
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_3Relu	BiasAdd_3*0
_output_shapes
:����������*
T0
�
	MaxPool_3MaxPoolRelu_3*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_4/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *

seed *(
_output_shapes
:��*
T0
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*(
_output_shapes
:��*
T0
�

Variable_8
VariableV2*
dtype0*(
_output_shapes
:��*
	container *
shared_name *
shape:��
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
y
Variable_8/readIdentity
Variable_8*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
V
Const_4Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_9
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shared_name *
shape:�
�
Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
l
Variable_9/readIdentity
Variable_9*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
Conv2D_4Conv2DReshapeVariable_8/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_4Relu	BiasAdd_4*0
_output_shapes
:����������*
T0
�
	MaxPool_4MaxPoolRelu_4*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3	MaxPool_4concat/axis*

Tidx0*0
_output_shapes
:����������*
N*
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����R  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
Tshape0*
T0
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
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
dtype0*
seed2 *

seed *(
_output_shapes
:����������*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
:����������*
T0
i
truncated_normal_5/shapeConst*
dtype0*
_output_shapes
:*
valueB"R     
\
truncated_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *

seed *
_output_shapes
:	�*
T0
�
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
_output_shapes
:	�*
T0
�
Variable_10
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shared_name *
shape:	�
�
Variable_10/AssignAssignVariable_10truncated_normal_5*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
s
Variable_10/readIdentityVariable_10*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
T
Const_5Const*
dtype0*
_output_shapes
:*
valueB*���=
w
Variable_11
VariableV2*
dtype0*
_output_shapes
:*
	container *
shared_name *
shape:
�
Variable_11/AssignAssignVariable_11Const_5*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
n
Variable_11/readIdentityVariable_11*
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
MatMulMatMuldropout/mulVariable_10/read*
transpose_b( *
transpose_a( *'
_output_shapes
:���������*
T0
V
addAddMatMulVariable_11/read*'
_output_shapes
:���������*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
H
ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
J
Shape_1Shapeadd*
_output_shapes
:*
out_type0*
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

axis *
_output_shapes
:*
N*
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
_output_shapes
:*
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
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
l
	Reshape_2Reshapeaddconcat_1*0
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
Shape_2Shapey_*
_output_shapes
:*
out_type0*
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

axis *
_output_shapes
:*
N*
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
_output_shapes
:*
T0
d
concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*

Tidx0*
_output_shapes
:*
N*
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

axis *
_output_shapes
:*
N*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*#
_output_shapes
:���������*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
Tshape0*
T0
Q
Const_6Const*
dtype0*
_output_shapes
:*
valueB: 
^
MeanMean	Reshape_4Const_6*
	keep_dims( *

Tidx0*
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
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
out_type0*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:���������*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
out_type0*
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
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
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_10/read*
transpose_b(*
transpose_a( *(
_output_shapes
:����������*
T0
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	�*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*1
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
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
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
:����������*
Tshape0*
T0
\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
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
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3	MaxPool_4*2
_output_shapes 
:::::*
N*
out_type0*
T0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3gradients/concat_grad/ShapeN:4*2
_output_shapes 
:::::*
N
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_4Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:4gradients/concat_grad/ShapeN:4*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3^gradients/concat_grad/Slice_4
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*.
_class$
" loc:@gradients/concat_grad/Slice*
T0
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0
�
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_3*
T0
�
0gradients/concat_grad/tuple/control_dependency_4Identitygradients/concat_grad/Slice_4'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_4*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_4_grad/MaxPoolGradMaxPoolGradRelu_4	MaxPool_40gradients/concat_grad/tuple/control_dependency_4*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*0
_output_shapes
:����������*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*0
_output_shapes
:����������*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:����������*
T0
�
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_3_grad/MaxPoolGradRelu_3*0
_output_shapes
:����������*
T0
�
gradients/Relu_4_grad/ReluGradReluGrad$gradients/MaxPool_4_grad/MaxPoolGradRelu_4*0
_output_shapes
:����������*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*
T0
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
�
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*
T0
�
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read* 
_output_shapes
::*
N*
out_type0*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read* 
_output_shapes
::*
N*
out_type0*
T0
�
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter*
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
VariableV2*
	container *
_class
loc:@Variable*
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *
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
VariableV2*
	container *
_class
loc:@Variable*
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *
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
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable
�
Variable/Adam
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
}
Variable/Adam/readIdentityVariable/Adam*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
Variable/Adam_1/readIdentityVariable/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_1
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_1
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam/readIdentityVariable_2/Adam*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_2
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_3
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_4
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam/readIdentityVariable_4/Adam*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_4
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*(
_output_shapes
:��*
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
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_5
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_5
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_6
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
Variable_6/Adam/readIdentityVariable_6/Adam*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_6
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_7
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
v
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam_1
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_7
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_8
�
Variable_8/Adam
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_8
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
Variable_8/Adam/readIdentityVariable_8/Adam*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_8
�
Variable_8/Adam_1
VariableV2*
	container *
shared_name *
shape:��*
dtype0*(
_output_shapes
:��*
_class
loc:@Variable_8
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_9
�
Variable_9/Adam
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_9
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
v
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_9
�
Variable_9/Adam_1
VariableV2*
	container *
shared_name *
shape:�*
dtype0*
_output_shapes	
:�*
_class
loc:@Variable_9
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
z
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
"Variable_10/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_10
�
Variable_10/Adam
VariableV2*
	container *
shared_name *
shape:	�*
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_10
�
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
}
Variable_10/Adam/readIdentityVariable_10/Adam*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_10
�
Variable_10/Adam_1
VariableV2*
	container *
shared_name *
shape:	�*
dtype0*
_output_shapes
:	�*
_class
loc:@Variable_10
�
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11
�
Variable_11/Adam
VariableV2*
	container *
shared_name *
shape:*
dtype0*
_output_shapes
:*
_class
loc:@Variable_11
�
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
x
Variable_11/Adam/readIdentityVariable_11/Adam*
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11
�
Variable_11/Adam_1
VariableV2*
	container *
shared_name *
shape:*
dtype0*
_output_shapes
:*
_class
loc:@Variable_11
�
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
_output_shapes
:*
_class
loc:@Variable_11*
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
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
t
ArgMaxArgMaxaddArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
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
Const_7Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_7*
	keep_dims( *

Tidx0*
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
N"c�+�S:     ��%	p��)��AJ��
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514�
n
xPlaceholder*
dtype0*,
_output_shapes
:����������*!
shape:����������
e
y_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"����   ,     
m
ReshapeReshapexReshape/shape*0
_output_shapes
:����������*
Tshape0*
T0
o
truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *
T0*(
_output_shapes
:��*

seed 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*(
_output_shapes
:��*
T0
w
truncated_normalAddtruncated_normal/multruncated_normal/mean*(
_output_shapes
:��*
T0
�
Variable
VariableV2*
dtype0*
	container *(
_output_shapes
:��*
shared_name *
shape:��
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
s
Variable/readIdentityVariable*(
_output_shapes
:��*
_class
loc:@Variable*
T0
T
ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_1
VariableV2*
dtype0*
	container *
_output_shapes	
:�*
shared_name *
shape:�
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
l
Variable_1/readIdentity
Variable_1*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
}
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
P
ReluReluBiasAdd*0
_output_shapes
:����������*
T0
�
MaxPoolMaxPoolRelu*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *
T0*(
_output_shapes
:��*

seed 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*(
_output_shapes
:��*
T0
�

Variable_2
VariableV2*
dtype0*
	container *(
_output_shapes
:��*
shared_name *
shape:��
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
y
Variable_2/readIdentity
Variable_2*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
V
Const_1Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_3
VariableV2*
dtype0*
	container *
_output_shapes	
:�*
shared_name *
shape:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
l
Variable_3/readIdentity
Variable_3*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_1Relu	BiasAdd_1*0
_output_shapes
:����������*
T0
�
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *
T0*(
_output_shapes
:��*

seed 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*(
_output_shapes
:��*
T0
�

Variable_4
VariableV2*
dtype0*
	container *(
_output_shapes
:��*
shared_name *
shape:��
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
y
Variable_4/readIdentity
Variable_4*(
_output_shapes
:��*
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
	container *
_output_shapes	
:�*
shared_name *
shape:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_2Relu	BiasAdd_2*0
_output_shapes
:����������*
T0
�
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *
T0*(
_output_shapes
:��*

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*(
_output_shapes
:��*
T0
�

Variable_6
VariableV2*
dtype0*
	container *(
_output_shapes
:��*
shared_name *
shape:��
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
y
Variable_6/readIdentity
Variable_6*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
V
Const_3Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_7
VariableV2*
dtype0*
	container *
_output_shapes	
:�*
shared_name *
shape:�
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_3Relu	BiasAdd_3*0
_output_shapes
:����������*
T0
�
	MaxPool_3MaxPoolRelu_3*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
q
truncated_normal_4/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ,     �   
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
dtype0*
seed2 *
T0*(
_output_shapes
:��*

seed 
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*(
_output_shapes
:��*
T0
}
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*(
_output_shapes
:��*
T0
�

Variable_8
VariableV2*
dtype0*
	container *(
_output_shapes
:��*
shared_name *
shape:��
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
y
Variable_8/readIdentity
Variable_8*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
V
Const_4Const*
dtype0*
_output_shapes	
:�*
valueB�*���=
x

Variable_9
VariableV2*
dtype0*
	container *
_output_shapes	
:�*
shared_name *
shape:�
�
Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
l
Variable_9/readIdentity
Variable_9*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
Conv2D_4Conv2DReshapeVariable_8/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
data_formatNHWC*0
_output_shapes
:����������*
T0
T
Relu_4Relu	BiasAdd_4*0
_output_shapes
:����������*
T0
�
	MaxPool_4MaxPoolRelu_4*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3	MaxPool_4concat/axis*

Tidx0*0
_output_shapes
:����������*
N*
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"����R  
n
	Reshape_1ReshapeconcatReshape_1/shape*(
_output_shapes
:����������*
Tshape0*
T0
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
dropout/ShapeShape	Reshape_1*
_output_shapes
:*
out_type0*
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
dtype0*
seed2 *
T0*(
_output_shapes
:����������*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:����������*
T0
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:����������*
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
:����������*
T0
i
truncated_normal_5/shapeConst*
dtype0*
_output_shapes
:*
valueB"R     
\
truncated_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *
T0*
_output_shapes
:	�*

seed 
�
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
_output_shapes
:	�*
T0
t
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
_output_shapes
:	�*
T0
�
Variable_10
VariableV2*
dtype0*
	container *
_output_shapes
:	�*
shared_name *
shape:	�
�
Variable_10/AssignAssignVariable_10truncated_normal_5*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
s
Variable_10/readIdentityVariable_10*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
T
Const_5Const*
dtype0*
_output_shapes
:*
valueB*���=
w
Variable_11
VariableV2*
dtype0*
	container *
_output_shapes
:*
shared_name *
shape:
�
Variable_11/AssignAssignVariable_11Const_5*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
n
Variable_11/readIdentityVariable_11*
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
MatMulMatMuldropout/mulVariable_10/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
V
addAddMatMulVariable_11/read*'
_output_shapes
:���������*
T0
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
H
ShapeShapeadd*
_output_shapes
:*
out_type0*
T0
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
J
Shape_1Shapeadd*
_output_shapes
:*
out_type0*
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

axis *
_output_shapes
:*
N*
T0
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
_output_shapes
:*
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
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
l
	Reshape_2Reshapeaddconcat_1*0
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
Shape_2Shapey_*
_output_shapes
:*
out_type0*
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

axis *
_output_shapes
:*
N*
T0
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
_output_shapes
:*
T0
d
concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
O
concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*

Tidx0*
_output_shapes
:*
N*
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

axis *
_output_shapes
:*
N*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*#
_output_shapes
:���������*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
Tshape0*
T0
Q
Const_6Const*
dtype0*
_output_shapes
:*
valueB: 
^
MeanMean	Reshape_4Const_6*
	keep_dims( *

Tidx0*
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
gradients/Mean_grad/ShapeShape	Reshape_4*
_output_shapes
:*
out_type0*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:���������*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
_output_shapes
:*
out_type0*
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
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
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
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
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_10/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*1
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
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
�
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:����������*
Tshape0*
T0
c
gradients/dropout/div_grad/NegNeg	Reshape_1*(
_output_shapes
:����������*
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
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:����������*5
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
:����������*
Tshape0*
T0
\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
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
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3	MaxPool_4*2
_output_shapes 
:::::*
out_type0*
N*
T0
�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3gradients/concat_grad/ShapeN:4*2
_output_shapes 
:::::*
N
�
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
gradients/concat_grad/Slice_4Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:4gradients/concat_grad/ShapeN:4*
Index0*J
_output_shapes8
6:4������������������������������������*
T0
�
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3^gradients/concat_grad/Slice_4
�
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*.
_class$
" loc:@gradients/concat_grad/Slice*
T0
�
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_1*
T0
�
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_2*
T0
�
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_3*
T0
�
0gradients/concat_grad/tuple/control_dependency_4Identitygradients/concat_grad/Slice_4'^gradients/concat_grad/tuple/group_deps*0
_output_shapes
:����������*0
_class&
$"loc:@gradients/concat_grad/Slice_4*
T0
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
$gradients/MaxPool_4_grad/MaxPoolGradMaxPoolGradRelu_4	MaxPool_40gradients/concat_grad/tuple/control_dependency_4*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*0
_output_shapes
:����������*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*0
_output_shapes
:����������*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:����������*
T0
�
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_3_grad/MaxPoolGradRelu_3*0
_output_shapes
:����������*
T0
�
gradients/Relu_4_grad/ReluGradReluGrad$gradients/MaxPool_4_grad/MaxPoolGradRelu_4*0
_output_shapes
:����������*
T0
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*0
_output_shapes
:����������*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
T0
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*
T0
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
�
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*
T0
�
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
�
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*
T0
�
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
T0
�
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
�
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*0
_output_shapes
:����������*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*
T0
�
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
_output_shapes	
:�*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
T0
�
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read* 
_output_shapes
::*
out_type0*
N*
T0
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*0
_output_shapes
:����������*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*(
_output_shapes
:��*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read* 
_output_shapes
::*
out_type0*
N*
T0
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read* 
_output_shapes
::*
out_type0*
N*
T0
�
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read* 
_output_shapes
::*
out_type0*
N*
T0
�
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*
T0
�
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read* 
_output_shapes
::*
out_type0*
N*
T0
�
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
*
T0
�
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*0
_output_shapes
:����������*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*
T0
�
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*(
_output_shapes
:��*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter*
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
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: *
_class
loc:@Variable
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *
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
VariableV2*
	container *
shared_name *
shape: *
dtype0*
_output_shapes
: *
_class
loc:@Variable
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *
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
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable
�
Variable/Adam
VariableV2*
	container *
_class
loc:@Variable*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
}
Variable/Adam/readIdentityVariable/Adam*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
!Variable/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable
�
Variable/Adam_1
VariableV2*
	container *
_class
loc:@Variable*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
Variable/Adam_1/readIdentityVariable/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1
�
Variable_1/Adam
VariableV2*
	container *
_class
loc:@Variable_1*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_1
�
Variable_1/Adam_1
VariableV2*
	container *
_class
loc:@Variable_1*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
!Variable_2/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
	container *
_class
loc:@Variable_2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam/readIdentityVariable_2/Adam*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_2
�
Variable_2/Adam_1
VariableV2*
	container *
_class
loc:@Variable_2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam
VariableV2*
	container *
_class
loc:@Variable_3*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_3
�
Variable_3/Adam_1
VariableV2*
	container *
_class
loc:@Variable_3*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
!Variable_4/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_4
�
Variable_4/Adam
VariableV2*
	container *
_class
loc:@Variable_4*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam/readIdentityVariable_4/Adam*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_4
�
Variable_4/Adam_1
VariableV2*
	container *
_class
loc:@Variable_4*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*(
_output_shapes
:��*
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
VariableV2*
	container *
_class
loc:@Variable_5*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
VariableV2*
	container *
_class
loc:@Variable_5*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
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
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_6
�
Variable_6/Adam
VariableV2*
	container *
_class
loc:@Variable_6*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
Variable_6/Adam/readIdentityVariable_6/Adam*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_6
�
Variable_6/Adam_1
VariableV2*
	container *
_class
loc:@Variable_6*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam
VariableV2*
	container *
_class
loc:@Variable_7*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
v
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_7
�
Variable_7/Adam_1
VariableV2*
	container *
_class
loc:@Variable_7*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_8
�
Variable_8/Adam
VariableV2*
	container *
_class
loc:@Variable_8*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
Variable_8/Adam/readIdentityVariable_8/Adam*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*(
_output_shapes
:��*'
valueB��*    *
_class
loc:@Variable_8
�
Variable_8/Adam_1
VariableV2*
	container *
_class
loc:@Variable_8*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name 
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_9
�
Variable_9/Adam
VariableV2*
	container *
_class
loc:@Variable_9*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
v
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *
_class
loc:@Variable_9
�
Variable_9/Adam_1
VariableV2*
	container *
_class
loc:@Variable_9*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
z
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
"Variable_10/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_10
�
Variable_10/Adam
VariableV2*
	container *
_class
loc:@Variable_10*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
}
Variable_10/Adam/readIdentityVariable_10/Adam*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*
valueB	�*    *
_class
loc:@Variable_10
�
Variable_10/Adam_1
VariableV2*
	container *
_class
loc:@Variable_10*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11
�
Variable_11/Adam
VariableV2*
	container *
_class
loc:@Variable_11*
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
x
Variable_11/Adam/readIdentityVariable_11/Adam*
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11
�
Variable_11/Adam_1
VariableV2*
	container *
_class
loc:@Variable_11*
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_11*
T0
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
_output_shapes
:*
_class
loc:@Variable_11*
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
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable*
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_1*
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_2*
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_3*
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_4*
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_5*
T0
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_6*
T0
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_7*
T0
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *(
_output_shapes
:��*
_class
loc:@Variable_8*
T0
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes	
:�*
_class
loc:@Variable_9*
T0
�
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes
:	�*
_class
loc:@Variable_10*
T0
�
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes
:*
_class
loc:@Variable_11*
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
t
ArgMaxArgMaxaddArgMax/dimension*

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
Cast_1CastEqual*

SrcT0
*

DstT0*#
_output_shapes
:���������
Q
Const_7Const*
dtype0*
_output_shapes
:*
valueB: 
_
accuracyMeanCast_1Const_7*
	keep_dims( *

Tidx0*
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
N"".
	summaries!

cross_entropy:0
accuracy_1:0"
train_op

Adam"�
	variables��
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
Variable_11/Adam_1:0Variable_11/Adam_1/AssignVariable_11/Adam_1/read:02&Variable_11/Adam_1/Initializer/zeros:0"�
trainable_variables��
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
Variable_11:0Variable_11/AssignVariable_11/read:02	Const_5:0��54       ^3\	�g*��A*)

cross_entropy�(k?


accuracy_1ͳ
?�ͣl6       OW��	9��+��A2*)

cross_entropy�]?


accuracy_1�?��{ 6       OW��	C�-��Ad*)

cross_entropy�NU?


accuracy_11�?p��7       ���Y	2��/��A�*)

cross_entropy7�P?


accuracy_1A?8g�I7       ���Y	��1��A�*)

cross_entropy٥K?


accuracy_1�� ?Fy��7       ���Y	���3��A�*)

cross_entropy��D?


accuracy_1"�#?��Im7       ���Y	���5��A�*)

cross_entropy��@?


accuracy_1�%?�_�D7       ���Y	N@8��A�*)

cross_entropyl3=?


accuracy_1�=(?���07       ���Y	�G:��A�*)

cross_entropy|:?


accuracy_1�=(?H�lq7       ���Y	�	=��A�*)

cross_entropy507?


accuracy_1�a*?��Q7       ���Y	,$?��A�*)

cross_entropy��3?


accuracy_1�;.?TH�Q7       ���Y	�\A��A�*)

cross_entropyc5?


accuracy_1�;.?��!#7       ���Y	��&C��A�*)

cross_entropyt�3?


accuracy_1(�,?��7       ���Y	ǚ(E��A�*)

cross_entropy'�1?


accuracy_1/?"Ȳe7       ���Y	�KG��A�*)

cross_entropyh/?


accuracy_1M�/?�9I�7       ���Y	�7qI��A�*)

cross_entropy�P2?


accuracy_1(�,?.7       ���Y	��zK��A�*)

cross_entropy�+?


accuracy_1��3?��7       ���Y	~�M��A�*)

cross_entropy��*?


accuracy_1��1?`zY7       ���Y	!kO��A�*)

cross_entropyk�*?


accuracy_1ڃ2? >g7       ���Y	�1_Q��A�*)

cross_entropy%�&?


accuracy_1;1??�b7       ���Y	�9S��A�*)

cross_entropy�J%?


accuracy_1.^6?_��c7       ���Y	��<U��A�*)

cross_entropy��%?


accuracy_1��3?p���7       ���Y	��1W��A�*)

cross_entropyGI(?


accuracy_1��3?�o�7       ���Y	u_/Y��A�*)

cross_entropy�$?


accuracy_1]97?�DV7       ���Y	�4[��A�	*)

cross_entropy�]#?


accuracy_1��6?��&7       ���Y	��<]��A�	*)

cross_entropyqs%?


accuracy_1��6?��p�7       ���Y	�+R_��A�
*)

cross_entropy%�"?


accuracy_19:4?��77       ���Y	�Eda��A�
*)

cross_entropyL@ ?


accuracy_1�8?�.d�7       ���Y	�B}c��A�
*)

cross_entropy2q?


accuracy_1$�8?�ĥ7       ���Y	�e��A�*)

cross_entropy�I!?


accuracy_1�8:?
�L7       ���Y	��g��A�*)

cross_entropy�}?


accuracy_1$�8?8��-7       ���Y	���i��A�*)

cross_entropy�?


accuracy_1��8?M�l�7       ���Y	�	l��A�*)

cross_entropy�?


accuracy_1$�8?[�7       ���Y	�i"n��A�*)

cross_entropy"�?


accuracy_1��9?a��7       ���Y	W�1p��A�*)

cross_entropy1K?


accuracy_1��9?5�{O7       ���Y	0�=r��A�*)

cross_entropy*�?


accuracy_1S]9?�1U!7       ���Y	+uXt��A�*)

cross_entropy" ?


accuracy_1]97?�!Q�7       ���Y	�2fv��A�*)

cross_entropy"�?


accuracy_1�8:?�@��7       ���Y	��{x��A�*)

cross_entropy3??


accuracy_1I�;?�"Sd7       ���Y	��z��A�*)

cross_entropy~�?


accuracy_1�8?|��G7       ���Y	RB�|��A�*)

cross_entropy�B ?


accuracy_1h5?Z��m7       ���Y	�_�~��A�*)

cross_entropy��?


accuracy_1��8?CD�G7       ���Y	�L���A�*)

cross_entropy�S?


accuracy_1��;?�0�7       ���Y	fD���A�*)

cross_entropy
�?


accuracy_1I�;?mqG�7       ���Y	�S����A�*)

cross_entropy+?


accuracy_1��8?���x7       ���Y	(ﲇ��A�*)

cross_entropy�6?


accuracy_1x\<?����7       ���Y	�����A�*)

cross_entropy��?


accuracy_1I�;?�b�7       ���Y	�����A�*)

cross_entropy:�?


accuracy_1I�;?`$A 7       ���Y	�S5���A�*)

cross_entropy)?


accuracy_1�<?]{Â7       ���Y	�{���A�*)

cross_entropy�?


accuracy_1�<?0��7       ���Y	�����A�*)

cross_entropy"�?


accuracy_1?�=?�'�7       ���Y	؆&���A�*)

cross_entropy|f?


accuracy_1S]9?W �R7       ���Y	��^���A�*)

cross_entropy��?


accuracy_1I�;?�NѨ7       ���Y	=����A�*)

cross_entropy��?


accuracy_1S]9?,*7       ���Y	n�����A�*)

cross_entropy��?


accuracy_1�;?j:m7       ���Y	�精��A�*)

cross_entropy��?


accuracy_1�8:?θș7       ���Y	�����A�*)

cross_entropy�?


accuracy_1�:?Uޖ7       ���Y	������A�*)

cross_entropy?


accuracy_1I�;?=�s7       ���Y	wRΣ��A�*)

cross_entropyp�?


accuracy_1��7?� �7       ���Y	C|����A�*)

cross_entropy��?


accuracy_1�<?mr7       ���Y	*���A�*)

cross_entropy�n?


accuracy_1�<?XHU�7       ���Y	P���A�*)

cross_entropy`k?


accuracy_1��9?C�^z7       ���Y	�K���A�*)

cross_entropy��?


accuracy_1��;?0:c�7       ���Y	
�C���A�*)

cross_entropy&n?


accuracy_1�7=?�ꆩ7       ���Y	��H���A�*)

cross_entropy�?


accuracy_1I�;?N��7       ���Y	ЩB���A�*)

cross_entropy�?


accuracy_1I�;?���U7       ���Y	?+9���A�*)

cross_entropy�?


accuracy_1�8:?�{B!7       ���Y	�>���A�*)

cross_entropyl-?


accuracy_1�;?�fQ~7       ���Y	J;k���A�*)

cross_entropy=7>?


accuracy_1ڃ2?���7       ���Y	Ξ����A�*)

cross_entropy�)?


accuracy_1�8?Wy-7       ���Y	����A�*)

cross_entropy��'?


accuracy_1�8:?KI��7       ���Y	�1A���A�*)

cross_entropyt?&?


accuracy_1�8:?$
7       ���Y	'~���A�*)

cross_entropy��(?


accuracy_1�;?�P�77       ���Y	tE����A�*)

cross_entropy��'?


accuracy_1�:?"N"o7       ���Y	������A�*)

cross_entropy9�&?


accuracy_1��;??�� 7       ���Y	������A�*)

cross_entropy��$?


accuracy_1�;?|�7       ���Y	�����A�*)

cross_entropy��"?


accuracy_1��8?�O�G7       ���Y	K$����A�*)

cross_entropy�p ?


accuracy_1x\<?�	b7       ���Y	<����A�*)

cross_entropy�}?


accuracy_1�<?��g7       ���Y	T�#���A�*)

cross_entropy]#?


accuracy_1?�=?4 "7       ���Y	�Y���A�*)

cross_entropy��?


accuracy_1�<?<,��7       ���Y	�ru���A�*)

cross_entropy�x ?


accuracy_1n�>?}A�
7       ���Y	��z���A� *)

cross_entropy�@?


accuracy_1�>?��.;7       ���Y	��}���A� *)

cross_entropyn ?


accuracy_1�;?��s7       ���Y	 -����A� *)

cross_entropy	?


accuracy_1�7=?V���7       ���Y	�����A�!*)

cross_entropy�?


accuracy_1�7=?[u�97       ���Y	�o����A�!*)

cross_entropy�Z#?


accuracy_1�;?���7       ���Y	�)���A�!*)

cross_entropy�d?


accuracy_1I�;?�m�<7       ���Y	�#;���A�"*)

cross_entropyO`?


accuracy_1�7=?���7       ���Y	 d���A�"*)

cross_entropyV�?


accuracy_1I�;?)��7       ���Y	h*x���A�#*)

cross_entropy��?


accuracy_1x\<?�ܓ|7       ���Y	X
����A�#*)

cross_entropyY�?


accuracy_1�[??�M��7       ���Y	�&����A�#*)

cross_entropy�m?


accuracy_1x\<?���Q7       ���Y	������A�$*)

cross_entropy�1?


accuracy_1�[??QO�7       ���Y	d2~���A�$*)

cross_entropy_W?


accuracy_1x\<?Ǐ~7       ���Y	luk���A�%*)

cross_entropy��?


accuracy_1?�=?�aj�7       ���Y	��d���A�%*)

cross_entropy�?


accuracy_1�<?���7       ���Y	'�X���A�%*)

cross_entropy"�?


accuracy_1�>?gw�7       ���Y	,=<���A�&*)

cross_entropyf�?


accuracy_1�;?��.�7       ���Y	ڔN���A�&*)

cross_entropyhf?


accuracy_1��;?��7       ���Y	3;U���A�'*)

cross_entropy.?


accuracy_1I�;?�L%7       ���Y	��n���A�'*)

cross_entropy��?


accuracy_1I�;?[<n7       ���Y	$F����A�'*)

cross_entropy�?


accuracy_1�<?Vx�7       ���Y	a����A�(*)

cross_entropy�?


accuracy_1I�;?i�>�7       ���Y	0,���A�(*)

cross_entropy��?


accuracy_1�>?n��a7       ���Y	�d���A�)*)

cross_entropy3�?


accuracy_1�7=?��vG7       ���Y	�����A�)*)

cross_entropys�?


accuracy_1�>?�S�T7       ���Y	$D
��A�)*)

cross_entropyh/?


accuracy_1�>?���7       ���Y	�Q��A�**)

cross_entropy�4?


accuracy_1�<?�U�7       ���Y	��Z��A�**)

cross_entropyM?


accuracy_1�7=?�K��7       ���Y	2]\��A�**)

cross_entropy?M?


accuracy_1�<?^(��7       ���Y	�W���A�+*)

cross_entropy3�?


accuracy_1n�>?LR�7       ���Y	k���A�+*)

cross_entropy�Y?


accuracy_1n�>?�:	�7       ���Y	@����A�,*)

cross_entropy	b?


accuracy_1�[??)�8�7       ���Y	Bɵ��A�,*)

cross_entropyS�?


accuracy_1�<?R�|K7       ���Y	Z����A�,*)

cross_entropy��?


accuracy_1�>?�� \7       ���Y	;����A�-*)

cross_entropyө?


accuracy_1?�=?ݽ��7       ���Y	���A�-*)

cross_entropy�?


accuracy_1x\<?��37       ���Y	ԤG!��A�.*)

cross_entropy�z?


accuracy_14�??Y�K7       ���Y	�sj#��A�.*)

cross_entropy�?


accuracy_1�;?��Յ7       ���Y	�H�%��A�.*)

cross_entropyU�?


accuracy_1?�=?S��7       ���Y	��(��A�/*)

cross_entropy��?


accuracy_1?�=?R�LQ7       ���Y	��U*��A�/*)

cross_entropyG�?


accuracy_1x\<?7��7       ���Y	2J�,��A�0*)

cross_entropy�}?


accuracy_1�[??Ɣ&�7       ���Y	c��.��A�0*)

cross_entropy�<?


accuracy_1�6@?|��q7       ���Y	jj�0��A�0*)

cross_entropy�{?


accuracy_1n�>?�=R+7       ���Y	a��2��A�1*)

cross_entropyN8?


accuracy_1x\<?..M�7       ���Y	�x4��A�1*)

cross_entropyKY?


accuracy_1�>?:��7       ���Y	 ��6��A�2*)

cross_entropy��?


accuracy_1?�=?�0��7       ���Y	+ڍ8��A�2*)

cross_entropy�?


accuracy_1��;?���7       ���Y	�[�:��A�2*)

cross_entropyQ�?


accuracy_1��;?"e�7       ���Y	SW!=��A�3*)

cross_entropy-I?


accuracy_1�[??��7       ���Y	��\?��A�3*)

cross_entropy�?


accuracy_1x\<?���s7       ���Y	d�A��A�3*)

cross_entropy�{?


accuracy_1I�;?�w�t7       ���Y	{�C��A�4*)

cross_entropy^/?


accuracy_1�>?�S��7       ���Y	Hv�E��A�4*)

cross_entropy�o?


accuracy_1�7=?.��7       ���Y	D��G��A�5*)

cross_entropy�?


accuracy_1?�=?���h7       ���Y	5�I��A�5*)

cross_entropy�C?


accuracy_1��;?b���7       ���Y	`�"L��A�5*)

cross_entropy�?


accuracy_1�7=?��-7       ���Y	�rN��A�6*)

cross_entropy��?


accuracy_1�7=?WZ5(7       ���Y	R��P��A�6*)

cross_entropy�?


accuracy_1�>?>X!�7       ���Y	Y�BS��A�7*)

cross_entropy	?


accuracy_1n�>?j��7       ���Y	�8�U��A�7*)

cross_entropy��?


accuracy_14�??���7       ���Y	a��W��A�7*)

cross_entropy�?


accuracy_1?�=?=-@7       ���Y	7PtY��A�8*)

cross_entropyd�?


accuracy_1��;?�Hc�7       ���Y	�j_[��A�8*)

cross_entropy�?


accuracy_1�7=?v�H7       ���Y	�8]��A�9*)

cross_entropy��?


accuracy_1?�=?���7       ���Y	:�_��A�9*)

cross_entropy��!?


accuracy_1�<?�$x�7       ���Y	)^�`��A�9*)

cross_entropy�d?


accuracy_1�>?��h~