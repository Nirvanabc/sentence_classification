       ЃK"	  @яЊжAbrain.Event:2ф"Vь      ym	*:JяЊжA"ѕз
l
xPlaceholder*+
_output_shapes
:џџџџџџџџџd*
dtype0* 
shape:џџџџџџџџџd
e
y_Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
f
Reshape/shapeConst*%
valueB"џџџџ   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*/
_output_shapes
:џџџџџџџџџd*
T0
o
truncated_normal/shapeConst*%
valueB"   d      <   *
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
Ђ
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
seed2 *&
_output_shapes
:d<*
dtype0*
T0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:d<*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:d<*
T0

Variable
VariableV2*
shared_name *
	container *&
_output_shapes
:d<*
shape:d<*
dtype0
Ќ
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
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
ConstConst*
valueB<*ЭЬЬ=*
dtype0*
_output_shapes
:<
v

Variable_1
VariableV2*
shared_name *
	container *
_output_shapes
:<*
shape:<*
dtype0

Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
use_locking(*
_output_shapes
:<*
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
Й
Conv2DConv2DReshapeVariable/read*/
_output_shapes
:џџџџџџџџџd<*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
]
addAddConv2DVariable_1/read*/
_output_shapes
:џџџџџџџџџd<*
T0
K
ReluReluadd*/
_output_shapes
:џџџџџџџџџd<*
T0
Є
MaxPoolMaxPoolRelu*/
_output_shapes
:џџџџџџџџџ2<*
data_formatNHWC*
ksize
*
strides
*
paddingSAME*
T0
q
truncated_normal_1/shapeConst*%
valueB"   d   <   x   *
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
І
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
seed2 *&
_output_shapes
:d<x*
dtype0*
T0

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:d<x*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:d<x*
T0


Variable_2
VariableV2*
shared_name *
	container *&
_output_shapes
:d<x*
shape:d<x*
dtype0
Д
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
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
Const_1Const*
valueBx*ЭЬЬ=*
dtype0*
_output_shapes
:x
v

Variable_3
VariableV2*
shared_name *
	container *
_output_shapes
:x*
shape:x*
dtype0

Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
use_locking(*
_output_shapes
:x*
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
Н
Conv2D_1Conv2DMaxPoolVariable_2/read*/
_output_shapes
:џџџџџџџџџ2x*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
a
add_1AddConv2D_1Variable_3/read*/
_output_shapes
:џџџџџџџџџ2x*
T0
O
Relu_1Reluadd_1*/
_output_shapes
:џџџџџџџџџ2x*
T0
Ј
	MaxPool_1MaxPoolRelu_1*/
_output_shapes
:џџџџџџџџџx*
data_formatNHWC*
ksize
*
strides
*
paddingSAME*
T0
q
truncated_normal_2/shapeConst*%
valueB"   d   x   №   *
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
Ї
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
seed2 *'
_output_shapes
:dx№*
dtype0*
T0

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*'
_output_shapes
:dx№*
T0
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*'
_output_shapes
:dx№*
T0


Variable_4
VariableV2*
shared_name *
	container *'
_output_shapes
:dx№*
shape:dx№*
dtype0
Е
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0
x
Variable_4/readIdentity
Variable_4*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0
V
Const_2Const*
valueB№*ЭЬЬ=*
dtype0*
_output_shapes	
:№
x

Variable_5
VariableV2*
shared_name *
	container *
_output_shapes	
:№*
shape:№*
dtype0

Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0
Р
Conv2D_2Conv2D	MaxPool_1Variable_4/read*0
_output_shapes
:џџџџџџџџџ№*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
b
add_2AddConv2D_2Variable_5/read*0
_output_shapes
:џџџџџџџџџ№*
T0
P
Relu_2Reluadd_2*0
_output_shapes
:џџџџџџџџџ№*
T0
Љ
	MaxPool_2MaxPoolRelu_2*0
_output_shapes
:џџџџџџџџџ№*
data_formatNHWC*
ksize
*
strides
*
paddingSAME*
T0
i
truncated_normal_3/shapeConst*
valueB"`  ш  *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
 
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
seed2 * 
_output_shapes
:
р0ш*
dtype0*
T0

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev* 
_output_shapes
:
р0ш*
T0
u
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean* 
_output_shapes
:
р0ш*
T0


Variable_6
VariableV2*
shared_name *
	container * 
_output_shapes
:
р0ш*
shape:
р0ш*
dtype0
Ў
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0
q
Variable_6/readIdentity
Variable_6* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0
V
Const_3Const*
valueBш*ЭЬЬ=*
dtype0*
_output_shapes	
:ш
x

Variable_7
VariableV2*
shared_name *
	container *
_output_shapes	
:ш*
shape:ш*
dtype0

Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0
`
Reshape_1/shapeConst*
valueB"џџџџ`  *
dtype0*
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_2Reshape_1/shape*
Tshape0*(
_output_shapes
:џџџџџџџџџр0*
T0

MatMulMatMul	Reshape_1Variable_6/read*(
_output_shapes
:џџџџџџџџџш*
transpose_b( *
transpose_a( *
T0
X
add_3AddMatMulVariable_7/read*(
_output_shapes
:џџџџџџџџџш*
T0
H
Relu_3Reluadd_3*(
_output_shapes
:џџџџџџџџџш*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
S
dropout/ShapeShapeRelu_3*
out_type0*
_output_shapes
:*
T0
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

seed *
seed2 *(
_output_shapes
:џџџџџџџџџш*
dtype0*
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџш*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџш*
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
:џџџџџџџџџш*
T0
i
truncated_normal_4/shapeConst*
valueB"ш     *
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

seed *
seed2 *
_output_shapes
:	ш*
dtype0*
T0

truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	ш*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	ш*
T0


Variable_8
VariableV2*
shared_name *
	container *
_output_shapes
:	ш*
shape:	ш*
dtype0
­
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0
p
Variable_8/readIdentity
Variable_8*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0
T
Const_4Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
v

Variable_9
VariableV2*
shared_name *
	container *
_output_shapes
:*
shape:*
dtype0

Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
use_locking(*
_output_shapes
:*
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

MatMul_1MatMuldropout/mulVariable_8/read*'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
transpose_a( *
T0
Y
add_4AddMatMul_1Variable_9/read*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
J
ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
L
Shape_1Shapeadd_4*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
T0*
_output_shapes
:*
N*

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
T0*
_output_shapes
:*
Index0
b
concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*

Tidx0*
_output_shapes
:*
N*
T0
l
	Reshape_2Reshapeadd_4concat*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
T0*
_output_shapes
:*
N*

axis 
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
_output_shapes
:*
Index0
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
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
k
	Reshape_3Reshapey_concat_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*
T0*
_output_shapes
:*
N*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*#
_output_shapes
:џџџџџџџџџ*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
^
MeanMean	Reshape_4Const_5*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
`
cross_entropy/tagsConst*
valueB Bcross_entropy*
dtype0*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
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
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 

gradients/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *
T0

gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
	keep_dims( *
T0

gradients/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
_output_shapes
:*
T0
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
b
gradients/add_4_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
gradients/add_4_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Б
gradients/add_4_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
т
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/add_4_grad/Reshape*
T0
л
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*
T0
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*(
_output_shapes
:џџџџџџџџџш*
transpose_b(*
transpose_a( *
T0
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_4_grad/tuple/control_dependency*
_output_shapes
:	ш*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes
:	ш*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*#
_output_shapes
:џџџџџџџџџ*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*#
_output_shapes
:џџџџџџџџџ*
T0
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ы
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0
ё
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
:џџџџџџџџџ*
T0
Ь
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
_output_shapes
:*
T0
Л
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџш*
T0
`
gradients/dropout/div_grad/NegNegRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
_output_shapes
:*
T0
Ѓ
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
Л
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0
ё
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0

gradients/Relu_3_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
`
gradients/add_3_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_3_grad/Shape_1Const*
valueB:ш*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџш*
T0
Џ
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
Tshape0*
_output_shapes	
:ш*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
у
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџш*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0
м
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
_output_shapes	
:ш*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0
П
gradients/MatMul_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*(
_output_shapes
:џџџџџџџџџр0*
transpose_b(*
transpose_a( *
T0
Г
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_3_grad/tuple/control_dependency* 
_output_shapes
:
р0ш*
transpose_b( *
transpose_a(*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџр0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps* 
_output_shapes
:
р0ш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_2*
out_type0*
_output_shapes
:*
T0
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*0
_output_shapes
:џџџџџџџџџ№*
T0
ѕ
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_2 gradients/Reshape_1_grad/Reshape*0
_output_shapes
:џџџџџџџџџ№*
T0*
ksize
*
strides
*
paddingSAME*
data_formatNHWC

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:џџџџџџџџџ№*
T0
b
gradients/add_2_grad/ShapeShapeConv2D_2*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
valueB:№*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*0
_output_shapes
:џџџџџџџџџ№*
T0
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
_output_shapes	
:№*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ы
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*0
_output_shapes
:џџџџџџџџџ№*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
_output_shapes	
:№*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0

gradients/Conv2D_2_grad/ShapeNShapeN	MaxPool_1Variable_4/read*
out_type0* 
_output_shapes
::*
N*
T0
Ю
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
Ь
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1 gradients/Conv2D_2_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџx*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*'
_output_shapes
:dx№*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/Conv2D_2_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ2x*
T0*
ksize
*
strides
*
paddingSAME*
data_formatNHWC

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:џџџџџџџџџ2x*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_1_grad/Shape_1Const*
valueB:x*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ѕ
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ2x*
T0
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
_output_shapes
:x*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ъ
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ2x*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0
л
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
_output_shapes
:x*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0* 
_output_shapes
::*
N*
T0
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ2<*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:d<x*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџd<*
T0*
ksize
*
strides
*
paddingSAME*
data_formatNHWC

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџd<*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
valueB:<*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџd<*
T0
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes
:<*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџd<*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:<*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0* 
_output_shapes
::*
N*
T0
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0
Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
T0

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџd*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*&
_output_shapes
:d<*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0*
_class
loc:@Variable

beta1_power
VariableV2*
_output_shapes
: *
shape: *
	container *
_class
loc:@Variable*
shared_name *
dtype0
Ћ
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
beta2_power/initial_valueConst*
valueB
 *wО?*
_output_shapes
: *
dtype0*
_class
loc:@Variable

beta2_power
VariableV2*
_output_shapes
: *
shape: *
	container *
_class
loc:@Variable*
shared_name *
dtype0
Ћ
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
Ё
Variable/Adam/Initializer/zerosConst*%
valueBd<*    *&
_output_shapes
:d<*
dtype0*
_class
loc:@Variable
Ў
Variable/Adam
VariableV2*
shape:d<*
_class
loc:@Variable*
	container *&
_output_shapes
:d<*
dtype0*
shared_name 
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
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
Ѓ
!Variable/Adam_1/Initializer/zerosConst*%
valueBd<*    *&
_output_shapes
:d<*
dtype0*
_class
loc:@Variable
А
Variable/Adam_1
VariableV2*
shape:d<*
_class
loc:@Variable*
	container *&
_output_shapes
:d<*
dtype0*
shared_name 
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*&
_output_shapes
:d<*
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

!Variable_1/Adam/Initializer/zerosConst*
valueB<*    *
_output_shapes
:<*
dtype0*
_class
loc:@Variable_1

Variable_1/Adam
VariableV2*
shape:<*
_class
loc:@Variable_1*
	container *
_output_shapes
:<*
dtype0*
shared_name 
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:<*
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

#Variable_1/Adam_1/Initializer/zerosConst*
valueB<*    *
_output_shapes
:<*
dtype0*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
shape:<*
_class
loc:@Variable_1*
	container *
_output_shapes
:<*
dtype0*
shared_name 
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:<*
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
Ѕ
!Variable_2/Adam/Initializer/zerosConst*%
valueBd<x*    *&
_output_shapes
:d<x*
dtype0*
_class
loc:@Variable_2
В
Variable_2/Adam
VariableV2*
shape:d<x*
_class
loc:@Variable_2*
	container *&
_output_shapes
:d<x*
dtype0*
shared_name 
Э
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0

Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0
Ї
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBd<x*    *&
_output_shapes
:d<x*
dtype0*
_class
loc:@Variable_2
Д
Variable_2/Adam_1
VariableV2*
shape:d<x*
_class
loc:@Variable_2*
	container *&
_output_shapes
:d<x*
dtype0*
shared_name 
г
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:d<x*
_class
loc:@Variable_2*
T0

!Variable_3/Adam/Initializer/zerosConst*
valueBx*    *
_output_shapes
:x*
dtype0*
_class
loc:@Variable_3

Variable_3/Adam
VariableV2*
shape:x*
_class
loc:@Variable_3*
	container *
_output_shapes
:x*
dtype0*
shared_name 
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:x*
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

#Variable_3/Adam_1/Initializer/zerosConst*
valueBx*    *
_output_shapes
:x*
dtype0*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
shape:x*
_class
loc:@Variable_3*
	container *
_output_shapes
:x*
dtype0*
shared_name 
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:x*
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
Ї
!Variable_4/Adam/Initializer/zerosConst*&
valueBdx№*    *'
_output_shapes
:dx№*
dtype0*
_class
loc:@Variable_4
Д
Variable_4/Adam
VariableV2*
shape:dx№*
_class
loc:@Variable_4*
	container *'
_output_shapes
:dx№*
dtype0*
shared_name 
Ю
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0

Variable_4/Adam/readIdentityVariable_4/Adam*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0
Љ
#Variable_4/Adam_1/Initializer/zerosConst*&
valueBdx№*    *'
_output_shapes
:dx№*
dtype0*
_class
loc:@Variable_4
Ж
Variable_4/Adam_1
VariableV2*
shape:dx№*
_class
loc:@Variable_4*
	container *'
_output_shapes
:dx№*
dtype0*
shared_name 
д
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*'
_output_shapes
:dx№*
_class
loc:@Variable_4*
T0

!Variable_5/Adam/Initializer/zerosConst*
valueB№*    *
_output_shapes	
:№*
dtype0*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
shape:№*
_class
loc:@Variable_5*
	container *
_output_shapes	
:№*
dtype0*
shared_name 
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0

#Variable_5/Adam_1/Initializer/zerosConst*
valueB№*    *
_output_shapes	
:№*
dtype0*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
shape:№*
_class
loc:@Variable_5*
	container *
_output_shapes	
:№*
dtype0*
shared_name 
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes	
:№*
_class
loc:@Variable_5*
T0

!Variable_6/Adam/Initializer/zerosConst*
valueB
р0ш*    * 
_output_shapes
:
р0ш*
dtype0*
_class
loc:@Variable_6
І
Variable_6/Adam
VariableV2*
shape:
р0ш*
_class
loc:@Variable_6*
	container * 
_output_shapes
:
р0ш*
dtype0*
shared_name 
Ч
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0
{
Variable_6/Adam/readIdentityVariable_6/Adam* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0

#Variable_6/Adam_1/Initializer/zerosConst*
valueB
р0ш*    * 
_output_shapes
:
р0ш*
dtype0*
_class
loc:@Variable_6
Ј
Variable_6/Adam_1
VariableV2*
shape:
р0ш*
_class
loc:@Variable_6*
	container * 
_output_shapes
:
р0ш*
dtype0*
shared_name 
Э
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0

Variable_6/Adam_1/readIdentityVariable_6/Adam_1* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
T0

!Variable_7/Adam/Initializer/zerosConst*
valueBш*    *
_output_shapes	
:ш*
dtype0*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shape:ш*
_class
loc:@Variable_7*
	container *
_output_shapes	
:ш*
dtype0*
shared_name 
Т
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0
v
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0

#Variable_7/Adam_1/Initializer/zerosConst*
valueBш*    *
_output_shapes	
:ш*
dtype0*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
shape:ш*
_class
loc:@Variable_7*
	container *
_output_shapes	
:ш*
dtype0*
shared_name 
Ш
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes	
:ш*
_class
loc:@Variable_7*
T0

!Variable_8/Adam/Initializer/zerosConst*
valueB	ш*    *
_output_shapes
:	ш*
dtype0*
_class
loc:@Variable_8
Є
Variable_8/Adam
VariableV2*
shape:	ш*
_class
loc:@Variable_8*
	container *
_output_shapes
:	ш*
dtype0*
shared_name 
Ц
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0
z
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0

#Variable_8/Adam_1/Initializer/zerosConst*
valueB	ш*    *
_output_shapes
:	ш*
dtype0*
_class
loc:@Variable_8
І
Variable_8/Adam_1
VariableV2*
shape:	ш*
_class
loc:@Variable_8*
	container *
_output_shapes
:	ш*
dtype0*
shared_name 
Ь
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes
:	ш*
_class
loc:@Variable_8*
T0

!Variable_9/Adam/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*
_class
loc:@Variable_9

Variable_9/Adam
VariableV2*
shape:*
_class
loc:@Variable_9*
	container *
_output_shapes
:*
dtype0*
shared_name 
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
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

#Variable_9/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*
_class
loc:@Variable_9

Variable_9/Adam_1
VariableV2*
shape:*
_class
loc:@Variable_9*
	container *
_output_shapes
:*
dtype0*
shared_name 
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
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
к
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:d<*
_class
loc:@Variable*
use_nesterov( 
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:<*
_class
loc:@Variable_1*
use_nesterov( 
ц
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_locking( *&
_output_shapes
:d<x*
_class
loc:@Variable_2*
use_nesterov( 
з
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:x*
_class
loc:@Variable_3*
use_nesterov( 
ч
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_locking( *'
_output_shapes
:dx№*
_class
loc:@Variable_4*
use_nesterov( 
и
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes	
:№*
_class
loc:@Variable_5*
use_nesterov( 
о
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( * 
_output_shapes
:
р0ш*
_class
loc:@Variable_6*
use_nesterov( 
и
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes	
:ш*
_class
loc:@Variable_7*
use_nesterov( 
п
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:	ш*
_class
loc:@Variable_8*
use_nesterov( 
з
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*
_class
loc:@Variable_9*
use_nesterov( 
Ч
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0
Щ

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable*
T0

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMaxArgMaxadd_4ArgMax/dimension*
output_type0	*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
_
accuracyMeanCast_1Const_6*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
dtype0*
_output_shapes
: 
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
: "Ш&ќq     `Eв	яЌ\яЊжAJф
Ю$Ќ$
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514ѕз
l
xPlaceholder* 
shape:џџџџџџџџџd*+
_output_shapes
:џџџџџџџџџd*
dtype0
e
y_Placeholder*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ*
dtype0
f
Reshape/shapeConst*%
valueB"џџџџ   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*/
_output_shapes
:џџџџџџџџџd*
T0
o
truncated_normal/shapeConst*%
valueB"   d      <   *
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
Ђ
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
T0*&
_output_shapes
:d<*

seed *
seed2 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:d<*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:d<*
T0

Variable
VariableV2*
shape:d<*
	container *
dtype0*
shared_name *&
_output_shapes
:d<
Ќ
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:d<*
_class
loc:@Variable
R
ConstConst*
valueB<*ЭЬЬ=*
dtype0*
_output_shapes
:<
v

Variable_1
VariableV2*
shape:<*
	container *
dtype0*
shared_name *
_output_shapes
:<

Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
T0*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:<*
_class
loc:@Variable_1
Й
Conv2DConv2DReshapeVariable/read*/
_output_shapes
:џџџџџџџџџd<*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
]
addAddConv2DVariable_1/read*/
_output_shapes
:џџџџџџџџџd<*
T0
K
ReluReluadd*/
_output_shapes
:џџџџџџџџџd<*
T0
Є
MaxPoolMaxPoolRelu*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ2<
q
truncated_normal_1/shapeConst*%
valueB"   d   <   x   *
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
І
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
T0*&
_output_shapes
:d<x*

seed *
seed2 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*&
_output_shapes
:d<x*
T0
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*&
_output_shapes
:d<x*
T0


Variable_2
VariableV2*
shape:d<x*
	container *
dtype0*
shared_name *&
_output_shapes
:d<x
Д
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*
T0*&
_output_shapes
:d<x*
_class
loc:@Variable_2
T
Const_1Const*
valueBx*ЭЬЬ=*
dtype0*
_output_shapes
:x
v

Variable_3
VariableV2*
shape:x*
	container *
dtype0*
shared_name *
_output_shapes
:x

Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
T0*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
k
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes
:x*
_class
loc:@Variable_3
Н
Conv2D_1Conv2DMaxPoolVariable_2/read*/
_output_shapes
:џџџџџџџџџ2x*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
a
add_1AddConv2D_1Variable_3/read*/
_output_shapes
:џџџџџџџџџ2x*
T0
O
Relu_1Reluadd_1*/
_output_shapes
:џџџџџџџџџ2x*
T0
Ј
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџx
q
truncated_normal_2/shapeConst*%
valueB"   d   x   №   *
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
Ї
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
T0*'
_output_shapes
:dx№*

seed *
seed2 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*'
_output_shapes
:dx№*
T0
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*'
_output_shapes
:dx№*
T0


Variable_4
VariableV2*
shape:dx№*
	container *
dtype0*
shared_name *'
_output_shapes
:dx№
Е
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4
x
Variable_4/readIdentity
Variable_4*
T0*'
_output_shapes
:dx№*
_class
loc:@Variable_4
V
Const_2Const*
valueB№*ЭЬЬ=*
dtype0*
_output_shapes	
:№
x

Variable_5
VariableV2*
shape:№*
	container *
dtype0*
shared_name *
_output_shapes	
:№

Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:№*
_class
loc:@Variable_5
Р
Conv2D_2Conv2D	MaxPool_1Variable_4/read*0
_output_shapes
:џџџџџџџџџ№*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
b
add_2AddConv2D_2Variable_5/read*0
_output_shapes
:џџџџџџџџџ№*
T0
P
Relu_2Reluadd_2*0
_output_shapes
:џџџџџџџџџ№*
T0
Љ
	MaxPool_2MaxPoolRelu_2*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ№
i
truncated_normal_3/shapeConst*
valueB"`  ш  *
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
 
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
T0* 
_output_shapes
:
р0ш*

seed *
seed2 

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev* 
_output_shapes
:
р0ш*
T0
u
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean* 
_output_shapes
:
р0ш*
T0


Variable_6
VariableV2*
shape:
р0ш*
	container *
dtype0*
shared_name * 
_output_shapes
:
р0ш
Ў
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6
q
Variable_6/readIdentity
Variable_6*
T0* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6
V
Const_3Const*
valueBш*ЭЬЬ=*
dtype0*
_output_shapes	
:ш
x

Variable_7
VariableV2*
shape:ш*
	container *
dtype0*
shared_name *
_output_shapes	
:ш

Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_7
`
Reshape_1/shapeConst*
valueB"џџџџ`  *
dtype0*
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_2Reshape_1/shape*
Tshape0*(
_output_shapes
:џџџџџџџџџр0*
T0

MatMulMatMul	Reshape_1Variable_6/read*
T0*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b( 
X
add_3AddMatMulVariable_7/read*(
_output_shapes
:џџџџџџџџџш*
T0
H
Relu_3Reluadd_3*(
_output_shapes
:џџџџџџџџџш*
T0
N
	keep_probPlaceholder*
shape:*
_output_shapes
:*
dtype0
S
dropout/ShapeShapeRelu_3*
out_type0*
_output_shapes
:*
T0
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
T0*(
_output_shapes
:џџџџџџџџџш*

seed *
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџш*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџш*
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
:џџџџџџџџџш*
T0
i
truncated_normal_4/shapeConst*
valueB"ш     *
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
_output_shapes
:	ш*

seed *
seed2 

truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
_output_shapes
:	ш*
T0
t
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
_output_shapes
:	ш*
T0


Variable_8
VariableV2*
shape:	ш*
	container *
dtype0*
shared_name *
_output_shapes
:	ш
­
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8
p
Variable_8/readIdentity
Variable_8*
T0*
_output_shapes
:	ш*
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
VariableV2*
shape:*
	container *
dtype0*
shared_name *
_output_shapes
:

Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
T0*
use_locking(*
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

MatMul_1MatMuldropout/mulVariable_8/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Y
add_4AddMatMul_1Variable_9/read*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
J
ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
L
Shape_1Shapeadd_4*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
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

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
_output_shapes
:*
T0
b
concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
T0*

Tidx0*
_output_shapes
:*
N
l
	Reshape_2Reshapeadd_4concat*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
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
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
_output_shapes
:*
T0
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
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
k
	Reshape_3Reshapey_concat_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*

axis *
N*
_output_shapes
:*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*#
_output_shapes
:џџџџџџџџџ*
T0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
^
MeanMean	Reshape_4Const_5*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
`
cross_entropy/tagsConst*
valueB Bcross_entropy*
dtype0*
_output_shapes
: 
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
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
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 

gradients/Mean_grad/ConstConst*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
dtype0
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Const_1Const*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
dtype0
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
dtype0
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

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
_output_shapes
:*
T0
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
c
gradients/Reshape_2_grad/ShapeShapeadd_4*
out_type0*
_output_shapes
:*
T0
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
b
gradients/add_4_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
gradients/add_4_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Б
gradients/add_4_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
т
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/add_4_grad/Reshape
л
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*
T0*(
_output_shapes
:џџџџџџџџџш*
transpose_a( *
transpose_b(
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_4_grad/tuple/control_dependency*
T0*
_output_shapes
:	ш*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџш*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	ш*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*#
_output_shapes
:џџџџџџџџџ*
T0
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*#
_output_shapes
:џџџџџџџџџ*
T0
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
_output_shapes
:*
T0
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
_output_shapes
:*
T0

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
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
:џџџџџџџџџ*
T0
Ь
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
_output_shapes
:*
T0
Л
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџш*
T0
`
gradients/dropout/div_grad/NegNegRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
_output_shapes
:*
T0

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
_output_shapes
:*
T0
Ѓ
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
Л
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџш*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
ё
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1

gradients/Relu_3_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_3*(
_output_shapes
:џџџџџџџџџш*
T0
`
gradients/add_3_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_3_grad/Shape_1Const*
valueB:ш*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџш*
T0
Џ
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
Tshape0*
_output_shapes	
:ш*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
у
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџш*/
_class%
#!loc:@gradients/add_3_grad/Reshape
м
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*
_output_shapes	
:ш*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
П
gradients/MatMul_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
T0*(
_output_shapes
:џџџџџџџџџр0*
transpose_a( *
transpose_b(
Г
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_3_grad/tuple/control_dependency*
T0* 
_output_shapes
:
р0ш*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџр0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
р0ш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_2*
out_type0*
_output_shapes
:*
T0
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*0
_output_shapes
:џџџџџџџџџ№*
T0
ѕ
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_2 gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*0
_output_shapes
:џџџџџџџџџ№

gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*0
_output_shapes
:џџџџџџџџџ№*
T0
b
gradients/add_2_grad/ShapeShapeConv2D_2*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
valueB:№*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
І
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*0
_output_shapes
:џџџџџџџџџ№*
T0
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
_output_shapes	
:№*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ы
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџ№*/
_class%
#!loc:@gradients/add_2_grad/Reshape
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*
_output_shapes	
:№*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1

gradients/Conv2D_2_grad/ShapeNShapeN	MaxPool_1Variable_4/read*
T0*
out_type0* 
_output_shapes
::*
N
Ю
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
Ь
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1 gradients/Conv2D_2_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџx*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*'
_output_shapes
:dx№*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter

$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_10gradients/Conv2D_2_grad/tuple/control_dependency*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ2x

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*/
_output_shapes
:џџџџџџџџџ2x*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_1_grad/Shape_1Const*
valueB:x*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ѕ
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ2x*
T0
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
_output_shapes
:x*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ъ
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџ2x*/
_class%
#!loc:@gradients/add_1_grad/Reshape
л
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:x*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
T0*
out_type0* 
_output_shapes
::*
N
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџ2<*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*&
_output_shapes
:d<x*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
T0*
ksize
*
strides
*
paddingSAME*/
_output_shapes
:џџџџџџџџџd<

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџd<*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
valueB:<*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџd<*
T0
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes
:<*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџd<*-
_class#
!loc:@gradients/add_grad/Reshape
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:<*/
_class%
#!loc:@gradients/add_grad/Reshape_1

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
T0*
out_type0* 
_output_shapes
::*
N
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(
Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
data_formatNHWC*
strides
*
paddingSAME*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџd*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:d<*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
_output_shapes
: *
dtype0

beta1_power
VariableV2*
shape: *
_class
loc:@Variable*
	container *
dtype0*
shared_name *
_output_shapes
: 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
T0*
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
 *wО?*
_class
loc:@Variable*
_output_shapes
: *
dtype0

beta2_power
VariableV2*
shape: *
_class
loc:@Variable*
	container *
dtype0*
shared_name *
_output_shapes
: 
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
T0*
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
Ё
Variable/Adam/Initializer/zerosConst*%
valueBd<*    *
_class
loc:@Variable*&
_output_shapes
:d<*
dtype0
Ў
Variable/Adam
VariableV2*
shape:d<*
shared_name *
	container *
_class
loc:@Variable*
dtype0*&
_output_shapes
:d<
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:d<*
_class
loc:@Variable
Ѓ
!Variable/Adam_1/Initializer/zerosConst*%
valueBd<*    *
_class
loc:@Variable*&
_output_shapes
:d<*
dtype0
А
Variable/Adam_1
VariableV2*
shape:d<*
shared_name *
	container *
_class
loc:@Variable*
dtype0*&
_output_shapes
:d<
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:d<*
_class
loc:@Variable

!Variable_1/Adam/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_1*
_output_shapes
:<*
dtype0

Variable_1/Adam
VariableV2*
shape:<*
shared_name *
	container *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:<
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:<*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_1*
_output_shapes
:<*
dtype0

Variable_1/Adam_1
VariableV2*
shape:<*
shared_name *
	container *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:<
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:<*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:<*
_class
loc:@Variable_1
Ѕ
!Variable_2/Adam/Initializer/zerosConst*%
valueBd<x*    *
_class
loc:@Variable_2*&
_output_shapes
:d<x*
dtype0
В
Variable_2/Adam
VariableV2*
shape:d<x*
shared_name *
	container *
_class
loc:@Variable_2*
dtype0*&
_output_shapes
:d<x
Э
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*&
_output_shapes
:d<x*
_class
loc:@Variable_2
Ї
#Variable_2/Adam_1/Initializer/zerosConst*%
valueBd<x*    *
_class
loc:@Variable_2*&
_output_shapes
:d<x*
dtype0
Д
Variable_2/Adam_1
VariableV2*
shape:d<x*
shared_name *
	container *
_class
loc:@Variable_2*
dtype0*&
_output_shapes
:d<x
г
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*&
_output_shapes
:d<x*
_class
loc:@Variable_2

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*&
_output_shapes
:d<x*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
valueBx*    *
_class
loc:@Variable_3*
_output_shapes
:x*
dtype0

Variable_3/Adam
VariableV2*
shape:x*
shared_name *
	container *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:x
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes
:x*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueBx*    *
_class
loc:@Variable_3*
_output_shapes
:x*
dtype0

Variable_3/Adam_1
VariableV2*
shape:x*
shared_name *
	container *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:x
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:x*
_class
loc:@Variable_3
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes
:x*
_class
loc:@Variable_3
Ї
!Variable_4/Adam/Initializer/zerosConst*&
valueBdx№*    *
_class
loc:@Variable_4*'
_output_shapes
:dx№*
dtype0
Д
Variable_4/Adam
VariableV2*
shape:dx№*
shared_name *
	container *
_class
loc:@Variable_4*
dtype0*'
_output_shapes
:dx№
Ю
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4

Variable_4/Adam/readIdentityVariable_4/Adam*
T0*'
_output_shapes
:dx№*
_class
loc:@Variable_4
Љ
#Variable_4/Adam_1/Initializer/zerosConst*&
valueBdx№*    *
_class
loc:@Variable_4*'
_output_shapes
:dx№*
dtype0
Ж
Variable_4/Adam_1
VariableV2*
shape:dx№*
shared_name *
	container *
_class
loc:@Variable_4*
dtype0*'
_output_shapes
:dx№
д
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:dx№*
_class
loc:@Variable_4

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*'
_output_shapes
:dx№*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueB№*    *
_class
loc:@Variable_5*
_output_shapes	
:№*
dtype0

Variable_5/Adam
VariableV2*
shape:№*
shared_name *
	container *
_class
loc:@Variable_5*
dtype0*
_output_shapes	
:№
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes	
:№*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueB№*    *
_class
loc:@Variable_5*
_output_shapes	
:№*
dtype0

Variable_5/Adam_1
VariableV2*
shape:№*
shared_name *
	container *
_class
loc:@Variable_5*
dtype0*
_output_shapes	
:№
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:№*
_class
loc:@Variable_5
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes	
:№*
_class
loc:@Variable_5

!Variable_6/Adam/Initializer/zerosConst*
valueB
р0ш*    *
_class
loc:@Variable_6* 
_output_shapes
:
р0ш*
dtype0
І
Variable_6/Adam
VariableV2*
shape:
р0ш*
shared_name *
	container *
_class
loc:@Variable_6*
dtype0* 
_output_shapes
:
р0ш
Ч
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6
{
Variable_6/Adam/readIdentityVariable_6/Adam*
T0* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6

#Variable_6/Adam_1/Initializer/zerosConst*
valueB
р0ш*    *
_class
loc:@Variable_6* 
_output_shapes
:
р0ш*
dtype0
Ј
Variable_6/Adam_1
VariableV2*
shape:
р0ш*
shared_name *
	container *
_class
loc:@Variable_6*
dtype0* 
_output_shapes
:
р0ш
Э
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6

Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0* 
_output_shapes
:
р0ш*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
valueBш*    *
_class
loc:@Variable_7*
_output_shapes	
:ш*
dtype0

Variable_7/Adam
VariableV2*
shape:ш*
shared_name *
	container *
_class
loc:@Variable_7*
dtype0*
_output_shapes	
:ш
Т
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueBш*    *
_class
loc:@Variable_7*
_output_shapes	
:ш*
dtype0

Variable_7/Adam_1
VariableV2*
shape:ш*
shared_name *
	container *
_class
loc:@Variable_7*
dtype0*
_output_shapes	
:ш
Ш
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:ш*
_class
loc:@Variable_7
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_7

!Variable_8/Adam/Initializer/zerosConst*
valueB	ш*    *
_class
loc:@Variable_8*
_output_shapes
:	ш*
dtype0
Є
Variable_8/Adam
VariableV2*
shape:	ш*
shared_name *
	container *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:	ш
Ц
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8
z
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_8

#Variable_8/Adam_1/Initializer/zerosConst*
valueB	ш*    *
_class
loc:@Variable_8*
_output_shapes
:	ш*
dtype0
І
Variable_8/Adam_1
VariableV2*
shape:	ш*
shared_name *
	container *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:	ш
Ь
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	ш*
_class
loc:@Variable_8
~
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_8

!Variable_9/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_9*
_output_shapes
:*
dtype0

Variable_9/Adam
VariableV2*
shape:*
shared_name *
	container *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
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
_class
loc:@Variable_9*
_output_shapes
:*
dtype0

Variable_9/Adam_1
VariableV2*
shape:*
shared_name *
	container *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
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
к
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable*&
_output_shapes
:d<*
T0
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_1*
_output_shapes
:<*
T0
ц
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_2*&
_output_shapes
:d<x*
T0
з
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_3*
_output_shapes
:x*
T0
ч
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_4*'
_output_shapes
:dx№*
T0
и
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_5*
_output_shapes	
:№*
T0
о
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_6* 
_output_shapes
:
р0ш*
T0
и
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_7*
_output_shapes	
:ш*
T0
п
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_8*
_output_shapes
:	ш*
T0
з
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_class
loc:@Variable_9*
_output_shapes
:*
T0
Ч
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
T0*
use_locking( *
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
Adam/mul_1*
validate_shape(*
T0*
use_locking( *
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
v
ArgMaxArgMaxadd_4ArgMax/dimension*
output_type0	*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
_
accuracyMeanCast_1Const_6*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
Z
accuracy_1/tagsConst*
valueB B
accuracy_1*
dtype0*
_output_shapes
: 
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
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0"ќ
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
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0".
	summaries!

cross_entropy:0
accuracy_1:0жЮ34       ^3\	к}ЦёЊжA*)

cross_entropyЖнХD


accuracy_1ьQИ>№љJБ6       OWя	\ъ (ЋжA*)

cross_entropy~v D


accuracy_1)\?кІP!6       OWя	9F`ЋжA(*)

cross_entropyњMuC


accuracy_1)\?>пЎ6       OWя	ђ8НЋжA<*)

cross_entropymnJC


accuracy_1   ?мeэ6       OWя	рџ^бЋжAP*)

cross_entropy_|CC


accuracy_1ЎGс>жџ6       OWя	[B^ЌжAd*)

cross_entropyРжїB


accuracy_1RИ?ћж6       OWя	быUЌжAx*)

cross_entropyx7D


accuracy_1Уѕ(?mi7       чшЪY	іЌжA*)

cross_entropyю&C


accuracy_1ы>СF97       чшЪY	зьБлЌжA *)

cross_entropyDЧB


accuracy_1Уѕ(?јЊU17       чшЪY	/ўЧ­жAД*)

cross_entropyфХёB


accuracy_1И?5a4я7       чшЪY	nTтa­жAШ*)

cross_entropyФљрB


accuracy_1)\?yЩu17       чшЪY	хFД­жAм*)

cross_entropyБўB


accuracy_1q=
?ЗЋ97       чшЪY	Xд­жA№*)

cross_entropyИdC


accuracy_1)\?!№u