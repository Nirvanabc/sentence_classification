       ЃK"	   ъжAbrain.Event:2bЭЃЧ      zуЖ 	*ъжA"
l
xPlaceholder*
dtype0* 
shape:џџџџџџџџџd*+
_output_shapes
:џџџџџџџџџd
e
y_Placeholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
f
Reshape/shapeConst*%
valueB"џџџџ   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*
T0*/
_output_shapes
:џџџџџџџџџd
o
truncated_normal/shapeConst*%
valueB"   d      F   *
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
dtype0*
seed2 *

seed *
T0*&
_output_shapes
:dF

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dF
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dF

Variable
VariableV2*
dtype0*
shape:dF*
	container *&
_output_shapes
:dF*
shared_name 
Ќ
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*&
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:dF*
_class
loc:@Variable
R
ConstConst*
valueBF*ЭЬЬ=*
dtype0*
_output_shapes
:F
v

Variable_1
VariableV2*
dtype0*
shape:F*
	container *
_output_shapes
:F*
shared_name 

Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_output_shapes
:F*
T0*
use_locking(*
_class
loc:@Variable_1
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:F*
_class
loc:@Variable_1
Й
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџdF
]
addAddConv2DVariable_1/read*
T0*/
_output_shapes
:џџџџџџџџџdF
K
ReluReluadd*
T0*/
_output_shapes
:џџџџџџџџџdF
Є
MaxPoolMaxPoolRelu*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ2F
q
truncated_normal_1/shapeConst*%
valueB"   d   F      *
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
Ї
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:dF

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:dF
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:dF


Variable_2
VariableV2*
dtype0*
shape:dF*
	container *'
_output_shapes
:dF*
shared_name 
Е
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*'
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable_2
x
Variable_2/readIdentity
Variable_2*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2
V
Const_1Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:
x

Variable_3
VariableV2*
dtype0*
shape:*
	container *
_output_shapes	
:*
shared_name 

Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*
_class
loc:@Variable_3
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:*
_class
loc:@Variable_3
О
Conv2D_1Conv2DMaxPoolVariable_2/read*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ2
b
add_1AddConv2D_1Variable_3/read*
T0*0
_output_shapes
:џџџџџџџџџ2
P
Relu_1Reluadd_1*
T0*0
_output_shapes
:џџџџџџџџџ2
Љ
	MaxPool_1MaxPoolRelu_1*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ
i
truncated_normal_2/shapeConst*
valueB"А6  ш  *
dtype0*
_output_shapes
:
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
 
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
Аmш

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
Аmш
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
Аmш


Variable_4
VariableV2*
dtype0*
shape:
Аmш*
	container * 
_output_shapes
:
Аmш*
shared_name 
Ў
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(* 
_output_shapes
:
Аmш*
T0*
use_locking(*
_class
loc:@Variable_4
q
Variable_4/readIdentity
Variable_4*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
V
Const_2Const*
valueBш*ЭЬЬ=*
dtype0*
_output_shapes	
:ш
x

Variable_5
VariableV2*
dtype0*
shape:ш*
	container *
_output_shapes	
:ш*
shared_name 

Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_output_shapes	
:ш*
T0*
use_locking(*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5
`
Reshape_1/shapeConst*
valueB"џџџџА6  *
dtype0*
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџАm

MatMulMatMul	Reshape_1Variable_4/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџш
X
add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:џџџџџџџџџш
H
Relu_2Reluadd_2*
T0*(
_output_shapes
:џџџџџџџџџш
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
S
dropout/ShapeShapeRelu_2*
out_type0*
T0*
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
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:џџџџџџџџџш
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџш

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџш
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
L
dropout/divRealDivRelu_2	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџш
i
truncated_normal_3/shapeConst*
valueB"ш     *
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

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	ш

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	ш
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	ш


Variable_6
VariableV2*
dtype0*
shape:	ш*
	container *
_output_shapes
:	ш*
shared_name 
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_output_shapes
:	ш*
T0*
use_locking(*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6
T
Const_3Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
	container *
_output_shapes
:*
shared_name 

Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
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

MatMul_1MatMuldropout/mulVariable_6/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ
Y
add_3AddMatMul_1Variable_7/read*
T0*'
_output_shapes
:џџџџџџџџџ
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
J
ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
L
Shape_1Shapeadd_3*
out_type0*
T0*
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
N*
T0*
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
Slice/size*
Index0*
T0*
_output_shapes
:
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
N*
T0*

Tidx0*
_output_shapes
:
l
	Reshape_2Reshapeadd_3concat*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
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
N*
T0*
_output_shapes
:
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
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
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_1*
Tshape0*
T0*0
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
N*
T0*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:џџџџџџџџџ
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( *
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
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

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
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
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
­
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Б
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/add_3_grad/Reshape
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџш
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	ш
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
out_type0*
T0*#
_output_shapes
:џџџџџџџџџ
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:џџџџџџџџџ
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
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
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџш
`
gradients/dropout/div_grad/NegNegRelu_2*
T0*(
_output_shapes
:џџџџџџџџџш
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
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
gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:џџџџџџџџџш
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
valueB:ш*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџш
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:ш
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
у
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџш*/
_class%
#!loc:@gradients/add_2_grad/Reshape
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*
_output_shapes	
:ш*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
П
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџАm
Г
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
Аmш
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџАm*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
Аmш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџ
ѕ
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ2

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:џџџџџџџџџ2
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџ2
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ы
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџ2*/
_class%
#!loc:@gradients/add_1_grad/Reshape
м
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*
_output_shapes	
:*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
T0*
N* 
_output_shapes
::
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџ2F*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*'
_output_shapes
:dF*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџdF

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:џџџџџџџџџdF
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:F*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*/
_output_shapes
:џџџџџџџџџdF
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:F
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџdF*-
_class#
!loc:@gradients/add_grad/Reshape
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:F*/
_class%
#!loc:@gradients/add_grad/Reshape_1

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
T0*
N* 
_output_shapes
::
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
:dF*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
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
VariableV2*
dtype0*
shape: *
_output_shapes
: *
	container *
shared_name *
_class
loc:@Variable
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
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
VariableV2*
dtype0*
shape: *
_output_shapes
: *
	container *
shared_name *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
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
valueBdF*    *
dtype0*&
_output_shapes
:dF*
_class
loc:@Variable
Ў
Variable/Adam
VariableV2*
dtype0*
shape:dF*&
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:dF*
_class
loc:@Variable
Ѓ
!Variable/Adam_1/Initializer/zerosConst*%
valueBdF*    *
dtype0*&
_output_shapes
:dF*
_class
loc:@Variable
А
Variable/Adam_1
VariableV2*
dtype0*
shape:dF*&
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:dF*
_class
loc:@Variable

!Variable_1/Adam/Initializer/zerosConst*
valueBF*    *
dtype0*
_output_shapes
:F*
_class
loc:@Variable_1

Variable_1/Adam
VariableV2*
dtype0*
shape:F*
_output_shapes
:F*
	container *
shared_name *
_class
loc:@Variable_1
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
T0*
use_locking(*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:F*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/zerosConst*
valueBF*    *
dtype0*
_output_shapes
:F*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
dtype0*
shape:F*
_output_shapes
:F*
	container *
shared_name *
_class
loc:@Variable_1
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
T0*
use_locking(*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:F*
_class
loc:@Variable_1
Ї
!Variable_2/Adam/Initializer/zerosConst*&
valueBdF*    *
dtype0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Д
Variable_2/Adam
VariableV2*
dtype0*
shape:dF*'
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable_2
Ю
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*'
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable_2

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Љ
#Variable_2/Adam_1/Initializer/zerosConst*&
valueBdF*    *
dtype0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Ж
Variable_2/Adam_1
VariableV2*
dtype0*
shape:dF*'
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable_2
д
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*'
_output_shapes
:dF*
T0*
use_locking(*
_class
loc:@Variable_2

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*
_class
loc:@Variable_3

Variable_3/Adam
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
	container *
shared_name *
_class
loc:@Variable_3
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*
_class
loc:@Variable_3
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
	container *
shared_name *
_class
loc:@Variable_3
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*
_class
loc:@Variable_3
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:*
_class
loc:@Variable_3

!Variable_4/Adam/Initializer/zerosConst*
valueB
Аmш*    *
dtype0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
І
Variable_4/Adam
VariableV2*
dtype0*
shape:
Аmш* 
_output_shapes
:
Аmш*
	container *
shared_name *
_class
loc:@Variable_4
Ч
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
Аmш*
T0*
use_locking(*
_class
loc:@Variable_4
{
Variable_4/Adam/readIdentityVariable_4/Adam*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/zerosConst*
valueB
Аmш*    *
dtype0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
Ј
Variable_4/Adam_1
VariableV2*
dtype0*
shape:
Аmш* 
_output_shapes
:
Аmш*
	container *
shared_name *
_class
loc:@Variable_4
Э
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
Аmш*
T0*
use_locking(*
_class
loc:@Variable_4

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueBш*    *
dtype0*
_output_shapes	
:ш*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
dtype0*
shape:ш*
_output_shapes	
:ш*
	container *
shared_name *
_class
loc:@Variable_5
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ш*
T0*
use_locking(*
_class
loc:@Variable_5
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueBш*    *
dtype0*
_output_shapes	
:ш*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
dtype0*
shape:ш*
_output_shapes	
:ш*
	container *
shared_name *
_class
loc:@Variable_5
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:ш*
T0*
use_locking(*
_class
loc:@Variable_5
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5

!Variable_6/Adam/Initializer/zerosConst*
valueB	ш*    *
dtype0*
_output_shapes
:	ш*
_class
loc:@Variable_6
Є
Variable_6/Adam
VariableV2*
dtype0*
shape:	ш*
_output_shapes
:	ш*
	container *
shared_name *
_class
loc:@Variable_6
Ц
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	ш*
T0*
use_locking(*
_class
loc:@Variable_6
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6

#Variable_6/Adam_1/Initializer/zerosConst*
valueB	ш*    *
dtype0*
_output_shapes
:	ш*
_class
loc:@Variable_6
І
Variable_6/Adam_1
VariableV2*
dtype0*
shape:	ш*
_output_shapes
:	ш*
	container *
shared_name *
_class
loc:@Variable_6
Ь
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	ш*
T0*
use_locking(*
_class
loc:@Variable_6
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name *
_class
loc:@Variable_7
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name *
_class
loc:@Variable_7
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*
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
use_nesterov( *&
_output_shapes
:dF*
T0*
use_locking( *
_class
loc:@Variable
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:F*
T0*
use_locking( *
_class
loc:@Variable_1
ч
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *'
_output_shapes
:dF*
T0*
use_locking( *
_class
loc:@Variable_2
и
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
T0*
use_locking( *
_class
loc:@Variable_3
о
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
Аmш*
T0*
use_locking( *
_class
loc:@Variable_4
и
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ш*
T0*
use_locking( *
_class
loc:@Variable_5
п
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	ш*
T0*
use_locking( *
_class
loc:@Variable_6
з
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
T0*
use_locking( *
_class
loc:@Variable_7

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
T0*
use_locking( *
_class
loc:@Variable


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
T0*
use_locking( *
_class
loc:@Variable
Р
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMaxArgMaxadd_3ArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
R
Cast_1CastEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
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
accuracy_1*
_output_shapes
: *
N"ZВc@ц      p	v:ъжAJГЬ
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514
l
xPlaceholder*
dtype0* 
shape:џџџџџџџџџd*+
_output_shapes
:џџџџџџџџџd
e
y_Placeholder*
dtype0*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ
f
Reshape/shapeConst*%
valueB"џџџџ   d      *
dtype0*
_output_shapes
:
l
ReshapeReshapexReshape/shape*
Tshape0*
T0*/
_output_shapes
:џџџџџџџџџd
o
truncated_normal/shapeConst*%
valueB"   d      F   *
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
dtype0*
seed2 *

seed *
T0*&
_output_shapes
:dF

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:dF
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:dF

Variable
VariableV2*
dtype0*
shape:dF*
	container *&
_output_shapes
:dF*
shared_name 
Ќ
Variable/AssignAssignVariabletruncated_normal*&
_output_shapes
:dF*
_class
loc:@Variable*
T0*
use_locking(*
validate_shape(
q
Variable/readIdentityVariable*
T0*&
_output_shapes
:dF*
_class
loc:@Variable
R
ConstConst*
valueBF*ЭЬЬ=*
dtype0*
_output_shapes
:F
v

Variable_1
VariableV2*
dtype0*
shape:F*
	container *
_output_shapes
:F*
shared_name 

Variable_1/AssignAssign
Variable_1Const*
_output_shapes
:F*
_class
loc:@Variable_1*
T0*
use_locking(*
validate_shape(
k
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
:F*
_class
loc:@Variable_1
Й
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџdF
]
addAddConv2DVariable_1/read*
T0*/
_output_shapes
:џџџџџџџџџdF
K
ReluReluadd*
T0*/
_output_shapes
:џџџџџџџџџdF
Є
MaxPoolMaxPoolRelu*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ2F
q
truncated_normal_1/shapeConst*%
valueB"   d   F      *
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
Ї
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:dF

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:dF
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:dF


Variable_2
VariableV2*
dtype0*
shape:dF*
	container *'
_output_shapes
:dF*
shared_name 
Е
Variable_2/AssignAssign
Variable_2truncated_normal_1*'
_output_shapes
:dF*
_class
loc:@Variable_2*
T0*
use_locking(*
validate_shape(
x
Variable_2/readIdentity
Variable_2*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2
V
Const_1Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:
x

Variable_3
VariableV2*
dtype0*
shape:*
	container *
_output_shapes	
:*
shared_name 

Variable_3/AssignAssign
Variable_3Const_1*
_output_shapes	
:*
_class
loc:@Variable_3*
T0*
use_locking(*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:*
_class
loc:@Variable_3
О
Conv2D_1Conv2DMaxPoolVariable_2/read*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ2
b
add_1AddConv2D_1Variable_3/read*
T0*0
_output_shapes
:џџџџџџџџџ2
P
Relu_1Reluadd_1*
T0*0
_output_shapes
:џџџџџџџџџ2
Љ
	MaxPool_1MaxPoolRelu_1*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ
i
truncated_normal_2/shapeConst*
valueB"А6  ш  *
dtype0*
_output_shapes
:
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
 
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
Аmш

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
Аmш
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
Аmш


Variable_4
VariableV2*
dtype0*
shape:
Аmш*
	container * 
_output_shapes
:
Аmш*
shared_name 
Ў
Variable_4/AssignAssign
Variable_4truncated_normal_2* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4*
T0*
use_locking(*
validate_shape(
q
Variable_4/readIdentity
Variable_4*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
V
Const_2Const*
valueBш*ЭЬЬ=*
dtype0*
_output_shapes	
:ш
x

Variable_5
VariableV2*
dtype0*
shape:ш*
	container *
_output_shapes	
:ш*
shared_name 

Variable_5/AssignAssign
Variable_5Const_2*
_output_shapes	
:ш*
_class
loc:@Variable_5*
T0*
use_locking(*
validate_shape(
l
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5
`
Reshape_1/shapeConst*
valueB"џџџџА6  *
dtype0*
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџАm

MatMulMatMul	Reshape_1Variable_4/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџш
X
add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:џџџџџџџџџш
H
Relu_2Reluadd_2*
T0*(
_output_shapes
:џџџџџџџџџш
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
S
dropout/ShapeShapeRelu_2*
out_type0*
T0*
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
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:џџџџџџџџџш
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:џџџџџџџџџш

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:џџџџџџџџџш
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
L
dropout/divRealDivRelu_2	keep_prob*
T0*
_output_shapes
:
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:џџџџџџџџџш
i
truncated_normal_3/shapeConst*
valueB"ш     *
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

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	ш

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	ш
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	ш


Variable_6
VariableV2*
dtype0*
shape:	ш*
	container *
_output_shapes
:	ш*
shared_name 
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
_output_shapes
:	ш*
_class
loc:@Variable_6*
T0*
use_locking(*
validate_shape(
p
Variable_6/readIdentity
Variable_6*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6
T
Const_3Const*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
	container *
_output_shapes
:*
shared_name 

Variable_7/AssignAssign
Variable_7Const_3*
_output_shapes
:*
_class
loc:@Variable_7*
T0*
use_locking(*
validate_shape(
k
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes
:*
_class
loc:@Variable_7

MatMul_1MatMuldropout/mulVariable_6/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ
Y
add_3AddMatMul_1Variable_7/read*
T0*'
_output_shapes
:џџџџџџџџџ
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
J
ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
L
Shape_1Shapeadd_3*
out_type0*
T0*
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
N*
T0*
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
Slice/size*
Index0*
T0*
_output_shapes
:
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

Tidx0*
T0*
N*
_output_shapes
:
l
	Reshape_2Reshapeadd_3concat*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
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
N*
T0*
_output_shapes
:
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
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
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
k
	Reshape_3Reshapey_concat_1*
Tshape0*
T0*0
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
N*
T0*
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:џџџџџџџџџ
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( 

gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( 
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
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
Є
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
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
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
T0*
_output_shapes
:
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
­
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Б
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*'
_output_shapes
:џџџџџџџџџ*/
_class%
#!loc:@gradients/add_3_grad/Reshape
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџш
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	ш
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
out_type0*
T0*#
_output_shapes
:џџџџџџџџџ
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:џџџџџџџџџ
Ь
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/dropout/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
З
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
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
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџш
`
gradients/dropout/div_grad/NegNegRelu_2*
T0*(
_output_shapes
:џџџџџџџџџш
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
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
gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:џџџџџџџџџш
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
valueB:ш*
dtype0*
_output_shapes
:
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
Tshape0*
T0*(
_output_shapes
:џџџџџџџџџш
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:ш
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
у
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџш*/
_class%
#!loc:@gradients/add_2_grad/Reshape
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*
_output_shapes	
:ш*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
П
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџАm
Г
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
Аmш
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:џџџџџџџџџАm*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
у
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
Аmш*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџ
ѕ
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*0
_output_shapes
:џџџџџџџџџ2

gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:џџџџџџџџџ2
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
T0*
_output_shapes
:
g
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:џџџџџџџџџ2
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ы
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*0
_output_shapes
:џџџџџџџџџ2*/
_class%
#!loc:@gradients/add_1_grad/Reshape
м
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*
_output_shapes	
:*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
T0*
N* 
_output_shapes
::
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџ2F*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*'
_output_shapes
:dF*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
T0*
strides
*
paddingSAME*
ksize
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџdF

gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:џџџџџџџџџdF
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:F*
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*/
_output_shapes
:џџџџџџџџџdF
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:F
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*/
_output_shapes
:џџџџџџџџџdF*-
_class#
!loc:@gradients/add_grad/Reshape
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:F*/
_class%
#!loc:@gradients/add_grad/Reshape_1

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
T0*
N* 
_output_shapes
::
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
strides
*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
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
:dF*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
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
VariableV2*
dtype0*
shape: *
_output_shapes
: *
	container *
shared_name *
_class
loc:@Variable
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
_class
loc:@Variable*
T0*
use_locking(*
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
VariableV2*
dtype0*
shape: *
_output_shapes
: *
	container *
shared_name *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
_class
loc:@Variable*
T0*
use_locking(*
validate_shape(
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable
Ё
Variable/Adam/Initializer/zerosConst*%
valueBdF*    *
dtype0*&
_output_shapes
:dF*
_class
loc:@Variable
Ў
Variable/Adam
VariableV2*
dtype0*
shape:dF*&
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*&
_output_shapes
:dF*
_class
loc:@Variable*
T0*
use_locking(*
validate_shape(
{
Variable/Adam/readIdentityVariable/Adam*
T0*&
_output_shapes
:dF*
_class
loc:@Variable
Ѓ
!Variable/Adam_1/Initializer/zerosConst*%
valueBdF*    *
dtype0*&
_output_shapes
:dF*
_class
loc:@Variable
А
Variable/Adam_1
VariableV2*
dtype0*
shape:dF*&
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*&
_output_shapes
:dF*
_class
loc:@Variable*
T0*
use_locking(*
validate_shape(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*&
_output_shapes
:dF*
_class
loc:@Variable

!Variable_1/Adam/Initializer/zerosConst*
valueBF*    *
dtype0*
_output_shapes
:F*
_class
loc:@Variable_1

Variable_1/Adam
VariableV2*
dtype0*
shape:F*
_output_shapes
:F*
	container *
shared_name *
_class
loc:@Variable_1
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
_output_shapes
:F*
_class
loc:@Variable_1*
T0*
use_locking(*
validate_shape(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes
:F*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/zerosConst*
valueBF*    *
dtype0*
_output_shapes
:F*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
dtype0*
shape:F*
_output_shapes
:F*
	container *
shared_name *
_class
loc:@Variable_1
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
_output_shapes
:F*
_class
loc:@Variable_1*
T0*
use_locking(*
validate_shape(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes
:F*
_class
loc:@Variable_1
Ї
!Variable_2/Adam/Initializer/zerosConst*&
valueBdF*    *
dtype0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Д
Variable_2/Adam
VariableV2*
dtype0*
shape:dF*'
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable_2
Ю
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*'
_output_shapes
:dF*
_class
loc:@Variable_2*
T0*
use_locking(*
validate_shape(

Variable_2/Adam/readIdentityVariable_2/Adam*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Љ
#Variable_2/Adam_1/Initializer/zerosConst*&
valueBdF*    *
dtype0*'
_output_shapes
:dF*
_class
loc:@Variable_2
Ж
Variable_2/Adam_1
VariableV2*
dtype0*
shape:dF*'
_output_shapes
:dF*
	container *
shared_name *
_class
loc:@Variable_2
д
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*'
_output_shapes
:dF*
_class
loc:@Variable_2*
T0*
use_locking(*
validate_shape(

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*'
_output_shapes
:dF*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*
_class
loc:@Variable_3

Variable_3/Adam
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
	container *
shared_name *
_class
loc:@Variable_3
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
_output_shapes	
:*
_class
loc:@Variable_3*
T0*
use_locking(*
validate_shape(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes	
:*
	container *
shared_name *
_class
loc:@Variable_3
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
_output_shapes	
:*
_class
loc:@Variable_3*
T0*
use_locking(*
validate_shape(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:*
_class
loc:@Variable_3

!Variable_4/Adam/Initializer/zerosConst*
valueB
Аmш*    *
dtype0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
І
Variable_4/Adam
VariableV2*
dtype0*
shape:
Аmш* 
_output_shapes
:
Аmш*
	container *
shared_name *
_class
loc:@Variable_4
Ч
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4*
T0*
use_locking(*
validate_shape(
{
Variable_4/Adam/readIdentityVariable_4/Adam*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/zerosConst*
valueB
Аmш*    *
dtype0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4
Ј
Variable_4/Adam_1
VariableV2*
dtype0*
shape:
Аmш* 
_output_shapes
:
Аmш*
	container *
shared_name *
_class
loc:@Variable_4
Э
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4*
T0*
use_locking(*
validate_shape(

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0* 
_output_shapes
:
Аmш*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueBш*    *
dtype0*
_output_shapes	
:ш*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
dtype0*
shape:ш*
_output_shapes	
:ш*
	container *
shared_name *
_class
loc:@Variable_5
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
_output_shapes	
:ш*
_class
loc:@Variable_5*
T0*
use_locking(*
validate_shape(
v
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueBш*    *
dtype0*
_output_shapes	
:ш*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
dtype0*
shape:ш*
_output_shapes	
:ш*
	container *
shared_name *
_class
loc:@Variable_5
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
_output_shapes	
:ш*
_class
loc:@Variable_5*
T0*
use_locking(*
validate_shape(
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes	
:ш*
_class
loc:@Variable_5

!Variable_6/Adam/Initializer/zerosConst*
valueB	ш*    *
dtype0*
_output_shapes
:	ш*
_class
loc:@Variable_6
Є
Variable_6/Adam
VariableV2*
dtype0*
shape:	ш*
_output_shapes
:	ш*
	container *
shared_name *
_class
loc:@Variable_6
Ц
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
_output_shapes
:	ш*
_class
loc:@Variable_6*
T0*
use_locking(*
validate_shape(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6

#Variable_6/Adam_1/Initializer/zerosConst*
valueB	ш*    *
dtype0*
_output_shapes
:	ш*
_class
loc:@Variable_6
І
Variable_6/Adam_1
VariableV2*
dtype0*
shape:	ш*
_output_shapes
:	ш*
	container *
shared_name *
_class
loc:@Variable_6
Ь
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_output_shapes
:	ш*
_class
loc:@Variable_6*
T0*
use_locking(*
validate_shape(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_output_shapes
:	ш*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name *
_class
loc:@Variable_7
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
_output_shapes
:*
_class
loc:@Variable_7*
T0*
use_locking(*
validate_shape(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name *
_class
loc:@Variable_7
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
_output_shapes
:*
_class
loc:@Variable_7*
T0*
use_locking(*
validate_shape(
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
use_nesterov( *&
_output_shapes
:dF*
T0*
use_locking( *
_class
loc:@Variable
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:F*
T0*
use_locking( *
_class
loc:@Variable_1
ч
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *'
_output_shapes
:dF*
T0*
use_locking( *
_class
loc:@Variable_2
и
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
T0*
use_locking( *
_class
loc:@Variable_3
о
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
Аmш*
T0*
use_locking( *
_class
loc:@Variable_4
и
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ш*
T0*
use_locking( *
_class
loc:@Variable_5
п
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	ш*
T0*
use_locking( *
_class
loc:@Variable_6
з
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
T0*
use_locking( *
_class
loc:@Variable_7

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
_class
loc:@Variable*
T0*
use_locking( *
validate_shape(


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
_class
loc:@Variable*
T0*
use_locking( *
validate_shape(
Р
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMaxArgMaxadd_3ArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
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
accuracy_1*
_output_shapes
: *
N""З
	variablesЉІ
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0".
	summaries!

cross_entropy:0
accuracy_1:0"Х
trainable_variables­Њ
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
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0"
train_op

Adam`3x4       ^3\	 OЇъжA*)

cross_entropyмБ	C


accuracy_1ЎGс>Чъ>6       OWя	rV$ъжA
*)

cross_entropyЧC


accuracy_1%I? чеф6       OWя	Zђ8ъжA*)

cross_entropyмB


accuracy_1Тѕ>цв66       OWя	й:еMъжA*)

cross_entropyЯqB


accuracy_1ы>AрВЧ6       OWя	щЌЩbъжA(*)

cross_entropy/<B


accuracy_1q=
?]Бc6       OWя	@ЏwъжA2*)

cross_entropytЊБB


accuracy_1=
з>MzЋВ6       OWя	dВІъжA<*)

cross_entropyЈiB


accuracy_1?5се6       OWя	нЎЁъжAF*)

cross_entropyћB


accuracy_1Єp=?\q66       OWя	SЗЖъжAP*)

cross_entropyф8ШA


accuracy_1Уѕ(?CYМ6       OWя	ВЫъжAZ*)

cross_entropyLB


accuracy_1q=
?ю%6       OWя	орЪръжAd*)

cross_entropyЦBB


accuracy_1   ?№?&@6       OWя	§ьѕъжAn*)

cross_entropy]ЛB


accuracy_1?

S6       OWя	Ўqс
ыжAx*)

cross_entropyh.ьA


accuracy_1{.?Фp	G7       чшЪY	)vвыжA*)

cross_entropyUёB


accuracy_1И?\"їў7       чшЪY	н!Щ4ыжA*)

cross_entropyЁA


accuracy_1RИ?ПЉ№7       чшЪY	ќдIыжA*)

cross_entropy­2A


accuracy_1333?ZБ7       чшЪY	йqб^ыжA *)

cross_entropy	УЄB


accuracy_1q=
?<б[7       чшЪY	кOуsыжAЊ*)

cross_entropyлЪA


accuracy_1RИ?ђ7       чшЪY	лмыжAД*)

cross_entropyЌvЉA


accuracy_1333?Нs7       чшЪY	№выжAО*)

cross_entropy_C


accuracy_1{.?nM7       чшЪY	бЙВыжAШ*)

cross_entropy/ыA


accuracy_1RИ?жС\7       чшЪY	gМшЧыжAв*)

cross_entropyУЂ^A


accuracy_1{.?bНё7       чшЪY	SБамыжAм*)

cross_entropyCвA


accuracy_1
з#?тзX7       чшЪY	@еёыжAц*)

cross_entropyoВA


accuracy_1\B?l;R7       чшЪY	аьжA№*)

cross_entropyИЃrA


accuracy_1{.?tЏ7       чшЪY	НРьжAњ*)

cross_entropyi>A


accuracy_1Уѕ(?о_Я7       чшЪY	sА0ьжA*)

cross_entropyhA


accuracy_1Уѕ(?эI_7       чшЪY	@бЌEьжA*)

cross_entropy`$A


accuracy_1{.?Дuk7       чшЪY	ЬЃZьжA*)

cross_entropyдЛA


accuracy_1RИ?P5м