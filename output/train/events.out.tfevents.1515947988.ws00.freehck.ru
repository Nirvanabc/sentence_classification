       ЃK"	   ѕсжAbrain.Event:27x№      Pв7f	/ѕсжA"ыр
n
xPlaceholder*
dtype0*,
_output_shapes
:џџџџџџџџџЌ*!
shape:џџџџџџџџџЌ
e
y_Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
f
Reshape/shapeConst*%
valueB"џџџџ   ,     *
dtype0*
_output_shapes
:
m
ReshapeReshapexReshape/shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџЌ
o
truncated_normal/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Є
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*(
_output_shapes
:ЌЊ
w
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*(
_output_shapes
:ЌЊ

Variable
VariableV2*
dtype0*(
_output_shapes
:ЌЊ*
shared_name *
shape:ЌЊ*
	container 
Ў
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
s
Variable/readIdentityVariable*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
T
ConstConst*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_1
VariableV2*
dtype0*
_output_shapes	
:Њ*
shared_name *
shape:Њ*
	container 

Variable_1/AssignAssign
Variable_1Const*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
l
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
Л
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
}
BiasAddBiasAddConv2DVariable_1/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
P
ReluReluBiasAdd*
T0*0
_output_shapes
:џџџџџџџџџЊ
І
MaxPoolMaxPoolRelu*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_1/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*(
_output_shapes
:ЌЊ


Variable_2
VariableV2*
dtype0*(
_output_shapes
:ЌЊ*
shared_name *
shape:ЌЊ*
	container 
Ж
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
y
Variable_2/readIdentity
Variable_2*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
V
Const_1Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_3
VariableV2*
dtype0*
_output_shapes	
:Њ*
shared_name *
shape:Њ*
	container 

Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
П
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_1MaxPoolRelu_1*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_2/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*(
_output_shapes
:ЌЊ


Variable_4
VariableV2*
dtype0*(
_output_shapes
:ЌЊ*
shared_name *
shape:ЌЊ*
	container 
Ж
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
y
Variable_4/readIdentity
Variable_4*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
V
Const_2Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_5
VariableV2*
dtype0*
_output_shapes	
:Њ*
shared_name *
shape:Њ*
	container 

Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
П
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_2Relu	BiasAdd_2*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_2MaxPoolRelu_2*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_3/shapeConst*%
valueB"   ,     Њ   *
dtype0*
_output_shapes
:
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*(
_output_shapes
:ЌЊ


Variable_6
VariableV2*
dtype0*(
_output_shapes
:ЌЊ*
shared_name *
shape:ЌЊ*
	container 
Ж
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
y
Variable_6/readIdentity
Variable_6*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
V
Const_3Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_7
VariableV2*
dtype0*
_output_shapes	
:Њ*
shared_name *
shape:Њ*
	container 

Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
П
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_3BiasAddConv2D_3Variable_7/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_3MaxPoolRelu_3*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3concat/axis*
T0*
N*

Tidx0*0
_output_shapes
:џџџџџџџџџЈ
`
Reshape_1/shapeConst*
valueB"џџџџP  *
dtype0*
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџа

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
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:џџџџџџџџџа

z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџа


dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџа

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
:џџџџџџџџџа

i
truncated_normal_4/shapeConst*
valueB"P     *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
T0*

seed *
seed2 *
_output_shapes
:	а


truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*
_output_shapes
:	а

t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*
_output_shapes
:	а



Variable_8
VariableV2*
dtype0*
_output_shapes
:	а
*
shared_name *
shape:	а
*
	container 
­
Variable_8/AssignAssign
Variable_8truncated_normal_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
p
Variable_8/readIdentity
Variable_8*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
T
Const_4Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
v

Variable_9
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
shape:*
	container 

Variable_9/AssignAssign
Variable_9Const_4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@Variable_9
k
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes
:*
_class
loc:@Variable_9

MatMulMatMuldropout/mulVariable_8/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:џџџџџџџџџ
U
addAddMatMulVariable_9/read*
T0*'
_output_shapes
:џџџџџџџџџ
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
Slice/beginPackSub*

axis *
T0*
N*
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
џџџџџџџџџ*
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
T0*
N*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
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
Slice_1/beginPackSub_1*

axis *
T0*
N*
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
џџџџџџџџџ*
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
T0*
N*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
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
Slice_2/sizePackSub_2*

axis *
T0*
N*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
^
MeanMean	Reshape_4Const_5*
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
 *  ?*
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

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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ
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

gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:џџџџџџџџџ
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
a
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*-
_class#
!loc:@gradients/add_grad/Reshape
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
Н
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:џџџџџџџџџа

В
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	а

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџа
*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
т
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	а
*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ы
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
ё
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
:џџџџџџџџџ
Ь
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
Л
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџа

c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:џџџџџџџџџа

}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
Ѓ
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
Л
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџа
*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
ё
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
Щ
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџЈ
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
 
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3*
N*
T0*
out_type0*,
_output_shapes
::::

"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*
N*,
_output_shapes
::::
ъ
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3
ы
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*.
_class$
" loc:@gradients/concat_grad/Slice
ё
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_1
ё
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_2
ё
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_3
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_3_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:џџџџџџџџџЊ

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
я
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
ш
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
ї
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
№
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad

$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
ї
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
№
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad

$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
ї
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
№
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
T0*
out_type0* 
_output_shapes
::
Ы
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Щ
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter

gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
N*
T0*
out_type0* 
_output_shapes
::
г
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter

gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N*
T0*
out_type0* 
_output_shapes
::
г
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter

gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read*
N*
T0*
out_type0* 
_output_shapes
::
г
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter

0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput

2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta1_power
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
_class
loc:@Variable*
shape: *
	container 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
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
 *wО?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta2_power
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
_class
loc:@Variable*
shape: *
	container 
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
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
Ѕ
Variable/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
В
Variable/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable
Ч
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
}
Variable/Adam/readIdentityVariable/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
Ї
!Variable/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
Д
Variable/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable
Э
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable

!Variable_1/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

Variable_1/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_1
Т
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
v
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_1
Ш
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
Љ
!Variable_2/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
Ж
Variable_2/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_2
Я
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
Ћ
#Variable_2/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
И
Variable_2/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_2
е
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

Variable_3/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_3
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_3
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
Љ
!Variable_4/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
Ж
Variable_4/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_4
Я
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4

Variable_4/Adam/readIdentityVariable_4/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
Ћ
#Variable_4/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
И
Variable_4/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_4
е
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_5
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_5
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
Љ
!Variable_6/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
Ж
Variable_6/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_6
Я
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6

Variable_6/Adam/readIdentityVariable_6/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
Ћ
#Variable_6/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
И
Variable_6/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
	container *
shape:ЌЊ*
_class
loc:@Variable_6
е
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6

Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_7
Т
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
	container *
shape:Њ*
_class
loc:@Variable_7
Ш
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

!Variable_8/Adam/Initializer/zerosConst*
valueB	а
*    *
dtype0*
_output_shapes
:	а
*
_class
loc:@Variable_8
Є
Variable_8/Adam
VariableV2*
shared_name *
_output_shapes
:	а
*
dtype0*
	container *
shape:	а
*
_class
loc:@Variable_8
Ц
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
z
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8

#Variable_8/Adam_1/Initializer/zerosConst*
valueB	а
*    *
dtype0*
_output_shapes
:	а
*
_class
loc:@Variable_8
І
Variable_8/Adam_1
VariableV2*
shared_name *
_output_shapes
:	а
*
dtype0*
	container *
shape:	а
*
_class
loc:@Variable_8
Ь
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8

!Variable_9/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_9

Variable_9/Adam
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
	container *
shape:*
_class
loc:@Variable_9
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@Variable_9
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_9

#Variable_9/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_9

Variable_9/Adam_1
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
	container *
shape:*
_class
loc:@Variable_9
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*
_class
loc:@Variable_9
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_9
W
Adam/learning_rateConst*
valueB
 *Зб8*
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
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
м
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *(
_output_shapes
:ЌЊ*
_class
loc:@Variable
к
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes	
:Њ*
_class
loc:@Variable_1
ш
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
м
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes	
:Њ*
_class
loc:@Variable_3
ш
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
м
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes	
:Њ*
_class
loc:@Variable_5
ш
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
м
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes	
:Њ*
_class
loc:@Variable_7
н
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:	а
*
_class
loc:@Variable_8
е
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:*
_class
loc:@Variable_9
Ч
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@Variable
Щ

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@Variable

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
output_type0	
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
output_type0	
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0

Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
_
accuracyMeanCast_1Const_6*
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
: ""бцe     І[Б	юН<ѕсжAJиІ
Ь'Њ'
9
Add
x"T
y"T
z"T"
Ttype:
2	
ы
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
Ш
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
ю
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
э
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

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
г
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
ы
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
2	

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
2	
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

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
2	
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

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
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514ыр
n
xPlaceholder*
dtype0*!
shape:џџџџџџџџџЌ*,
_output_shapes
:џџџџџџџџџЌ
e
y_Placeholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
f
Reshape/shapeConst*%
valueB"џџџџ   ,     *
dtype0*
_output_shapes
:
m
ReshapeReshapexReshape/shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџЌ
o
truncated_normal/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Є
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*(
_output_shapes
:ЌЊ
w
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*(
_output_shapes
:ЌЊ

Variable
VariableV2*
dtype0*
shape:ЌЊ*
shared_name *(
_output_shapes
:ЌЊ*
	container 
Ў
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*(
_output_shapes
:ЌЊ*
validate_shape(
s
Variable/readIdentityVariable*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
T
ConstConst*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_1
VariableV2*
dtype0*
shape:Њ*
shared_name *
_output_shapes	
:Њ*
	container 

Variable_1/AssignAssign
Variable_1Const*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:Њ*
validate_shape(
l
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
Л
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
}
BiasAddBiasAddConv2DVariable_1/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
P
ReluReluBiasAdd*
T0*0
_output_shapes
:џџџџџџџџџЊ
І
MaxPoolMaxPoolRelu*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_1/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*(
_output_shapes
:ЌЊ


Variable_2
VariableV2*
dtype0*
shape:ЌЊ*
shared_name *(
_output_shapes
:ЌЊ*
	container 
Ж
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*(
_output_shapes
:ЌЊ*
validate_shape(
y
Variable_2/readIdentity
Variable_2*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
V
Const_1Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_3
VariableV2*
dtype0*
shape:Њ*
shared_name *
_output_shapes	
:Њ*
	container 

Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Њ*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
П
Conv2D_1Conv2DReshapeVariable_2/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_1MaxPoolRelu_1*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_2/shapeConst*%
valueB"   ,     Њ   *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*(
_output_shapes
:ЌЊ


Variable_4
VariableV2*
dtype0*
shape:ЌЊ*
shared_name *(
_output_shapes
:ЌЊ*
	container 
Ж
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*(
_output_shapes
:ЌЊ*
validate_shape(
y
Variable_4/readIdentity
Variable_4*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
V
Const_2Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_5
VariableV2*
dtype0*
shape:Њ*
shared_name *
_output_shapes	
:Њ*
	container 

Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes	
:Њ*
validate_shape(
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
П
Conv2D_2Conv2DReshapeVariable_4/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_2BiasAddConv2D_2Variable_5/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_2Relu	BiasAdd_2*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_2MaxPoolRelu_2*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
q
truncated_normal_3/shapeConst*%
valueB"   ,     Њ   *
dtype0*
_output_shapes
:
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Ј
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:ЌЊ

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*(
_output_shapes
:ЌЊ
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*(
_output_shapes
:ЌЊ


Variable_6
VariableV2*
dtype0*
shape:ЌЊ*
shared_name *(
_output_shapes
:ЌЊ*
	container 
Ж
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*(
_output_shapes
:ЌЊ*
validate_shape(
y
Variable_6/readIdentity
Variable_6*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
V
Const_3Const*
valueBЊ*ЭЬЬ=*
dtype0*
_output_shapes	
:Њ
x

Variable_7
VariableV2*
dtype0*
shape:Њ*
shared_name *
_output_shapes	
:Њ*
	container 

Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
_class
loc:@Variable_7*
T0*
_output_shapes	
:Њ*
validate_shape(
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
П
Conv2D_3Conv2DReshapeVariable_6/read*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

	BiasAdd_3BiasAddConv2D_3Variable_7/read*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:џџџџџџџџџЊ
Њ
	MaxPool_3MaxPoolRelu_3*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concatConcatV2MaxPool	MaxPool_1	MaxPool_2	MaxPool_3concat/axis*
T0*
N*

Tidx0*0
_output_shapes
:џџџџџџџџџЈ
`
Reshape_1/shapeConst*
valueB"џџџџP  *
dtype0*
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџа

N
	keep_probPlaceholder*
dtype0*
shape:*
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
 *  ?*
dtype0*
_output_shapes
: 

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
T0*

seed *
seed2 *(
_output_shapes
:џџџџџџџџџа

z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџа


dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџа

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
:џџџџџџџџџа

i
truncated_normal_4/shapeConst*
valueB"P     *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 

"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
T0*

seed *
seed2 *
_output_shapes
:	а


truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*
_output_shapes
:	а

t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*
_output_shapes
:	а



Variable_8
VariableV2*
dtype0*
shape:	а
*
shared_name *
_output_shapes
:	а
*
	container 
­
Variable_8/AssignAssign
Variable_8truncated_normal_4*
use_locking(*
_class
loc:@Variable_8*
T0*
_output_shapes
:	а
*
validate_shape(
p
Variable_8/readIdentity
Variable_8*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
T
Const_4Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
v

Variable_9
VariableV2*
dtype0*
shape:*
shared_name *
_output_shapes
:*
	container 

Variable_9/AssignAssign
Variable_9Const_4*
use_locking(*
_class
loc:@Variable_9*
T0*
_output_shapes
:*
validate_shape(
k
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes
:*
_class
loc:@Variable_9

MatMulMatMuldropout/mulVariable_8/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:џџџџџџџџџ
U
addAddMatMulVariable_9/read*
T0*'
_output_shapes
:џџџџџџџџџ
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
Slice/beginPackSub*

axis *
T0*
N*
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
џџџџџџџџџ*
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
T0*
N*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
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
Slice_1/beginPackSub_1*

axis *
T0*
N*
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
џџџџџџџџџ*
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
T0*
N*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
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
Slice_2/sizePackSub_2*

axis *
T0*
N*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
^
MeanMean	Reshape_4Const_5*
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
 *  ?*
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

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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ
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

gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:џџџџџџџџџ
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
a
gradients/Reshape_2_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
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
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
к
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*-
_class#
!loc:@gradients/add_grad/Reshape
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
Н
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:џџџџџџџџџа

В
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	а

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџа
*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
т
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	а
*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ы
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
ё
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
:џџџџџџџџџ
Ь
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
Л
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџа

c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:џџџџџџџџџа

}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
Ѓ
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
Л
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџа
*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
ё
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
Щ
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџЈ
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
 
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2	MaxPool_3*
out_type0*
T0*
N*,
_output_shapes
::::

"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2gradients/concat_grad/ShapeN:3*
N*,
_output_shapes
::::
ъ
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
№
gradients/concat_grad/Slice_3Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:3gradients/concat_grad/ShapeN:3*
T0*
Index0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2^gradients/concat_grad/Slice_3
ы
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*.
_class$
" loc:@gradients/concat_grad/Slice
ё
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_1
ё
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_2
ё
0gradients/concat_grad/tuple/control_dependency_3Identitygradients/concat_grad/Slice_3'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*0
_class&
$"loc:@gradients/concat_grad/Slice_3
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool.gradients/concat_grad/tuple/control_dependency*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

$gradients/MaxPool_3_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_30gradients/concat_grad/tuple/control_dependency_3*
paddingVALID*
strides
*
ksize
*
T0*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
T0*0
_output_shapes
:џџџџџџџџџЊ

gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_3_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:џџџџџџџџџЊ

"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
я
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
ш
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad

$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
ї
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
№
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad

$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
ї
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
№
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad

$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:Њ
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
ї
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЊ*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
№
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*
_output_shapes	
:Њ*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
T0*
N* 
_output_shapes
::
Ы
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Щ
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter

gradients/Conv2D_1_grad/ShapeNShapeNReshapeVariable_2/read*
out_type0*
T0*
N* 
_output_shapes
::
г
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter

gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
out_type0*
T0*
N* 
_output_shapes
::
г
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter

gradients/Conv2D_3_grad/ShapeNShapeNReshapeVariable_6/read*
out_type0*
T0*
N* 
_output_shapes
::
г
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Я
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter

0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџЌ*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput

2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*(
_output_shapes
:ЌЊ*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta1_power
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
	container *
shape: *
_class
loc:@Variable
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
_output_shapes
: *
validate_shape(
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
 *wО?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta2_power
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
	container *
shape: *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
_output_shapes
: *
validate_shape(
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable
Ѕ
Variable/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
В
Variable/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable*
shape:ЌЊ*
	container 
Ч
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*(
_output_shapes
:ЌЊ*
validate_shape(
}
Variable/Adam/readIdentityVariable/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
Ї
!Variable/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
Д
Variable/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable*
shape:ЌЊ*
	container 
Э
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable

!Variable_1/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

Variable_1/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_1*
shape:Њ*
	container 
Т
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:Њ*
validate_shape(
v
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_1*
shape:Њ*
	container 
Ш
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
_output_shapes	
:Њ*
validate_shape(
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
Љ
!Variable_2/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
Ж
Variable_2/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_2*
shape:ЌЊ*
	container 
Я
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
Ћ
#Variable_2/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
И
Variable_2/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_2*
shape:ЌЊ*
	container 
е
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

Variable_3/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_3*
shape:Њ*
	container 
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Њ*
validate_shape(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_3*
shape:Њ*
	container 
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Њ*
validate_shape(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
Љ
!Variable_4/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
Ж
Variable_4/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_4*
shape:ЌЊ*
	container 
Я
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_4/Adam/readIdentityVariable_4/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
Ћ
#Variable_4/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
И
Variable_4/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_4*
shape:ЌЊ*
	container 
е
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_5*
shape:Њ*
	container 
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes	
:Њ*
validate_shape(
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_5*
shape:Њ*
	container 
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
_output_shapes	
:Њ*
validate_shape(
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
Љ
!Variable_6/Adam/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
Ж
Variable_6/Adam
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_6*
shape:ЌЊ*
	container 
Я
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_6/Adam/readIdentityVariable_6/Adam*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
Ћ
#Variable_6/Adam_1/Initializer/zerosConst*'
valueBЌЊ*    *
dtype0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
И
Variable_6/Adam_1
VariableV2*
shared_name *(
_output_shapes
:ЌЊ*
dtype0*
_class
loc:@Variable_6*
shape:ЌЊ*
	container 
е
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*(
_output_shapes
:ЌЊ*
validate_shape(

Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_7*
shape:Њ*
	container 
Т
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
_output_shapes	
:Њ*
validate_shape(
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueBЊ*    *
dtype0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
shared_name *
_output_shapes	
:Њ*
dtype0*
_class
loc:@Variable_7*
shape:Њ*
	container 
Ш
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
_output_shapes	
:Њ*
validate_shape(
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7

!Variable_8/Adam/Initializer/zerosConst*
valueB	а
*    *
dtype0*
_output_shapes
:	а
*
_class
loc:@Variable_8
Є
Variable_8/Adam
VariableV2*
shared_name *
_output_shapes
:	а
*
dtype0*
_class
loc:@Variable_8*
shape:	а
*
	container 
Ц
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_8*
T0*
_output_shapes
:	а
*
validate_shape(
z
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8

#Variable_8/Adam_1/Initializer/zerosConst*
valueB	а
*    *
dtype0*
_output_shapes
:	а
*
_class
loc:@Variable_8
І
Variable_8/Adam_1
VariableV2*
shared_name *
_output_shapes
:	а
*
dtype0*
_class
loc:@Variable_8*
shape:	а
*
	container 
Ь
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_8*
T0*
_output_shapes
:	а
*
validate_shape(
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8

!Variable_9/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_9

Variable_9/Adam
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
shape:*
	container 
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_9*
T0*
_output_shapes
:*
validate_shape(
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_9

#Variable_9/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_9

Variable_9/Adam_1
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
_class
loc:@Variable_9*
shape:*
	container 
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_9*
T0*
_output_shapes
:*
validate_shape(
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_9
W
Adam/learning_rateConst*
valueB
 *Зб8*
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
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
м
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable
к
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_1
ш
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_2
м
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_3
ш
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_4
м
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_5
ш
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*(
_output_shapes
:ЌЊ*
_class
loc:@Variable_6
м
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:Њ*
_class
loc:@Variable_7
н
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:	а
*
_class
loc:@Variable_8
е
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:*
_class
loc:@Variable_9
Ч
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_class
loc:@Variable*
T0*
_output_shapes
: *
validate_shape(
Щ

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_class
loc:@Variable*
T0*
_output_shapes
: *
validate_shape(

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
output_type0	
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
output_type0	
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0

Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
_
accuracyMeanCast_1Const_6*
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
accuracy_1:0"ќ
	variablesюы
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
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0"
train_op

Adam"в
trainable_variablesКЗ
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
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0mЮ