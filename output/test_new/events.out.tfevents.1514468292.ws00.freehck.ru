       ЃK"	   ё<жAbrain.Event:2MјЩ2НЧ      Ъ§+З	ї;ё<жA"А
l
xPlaceholder*+
_output_shapes
:џџџџџџџџџd* 
shape:џџџџџџџџџd*
dtype0
e
y_Placeholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
f
Reshape/shapeConst*
_output_shapes
:*%
valueB"џџџџ   d      *
dtype0
l
ReshapeReshapexReshape/shape*/
_output_shapes
:џџџџџџџџџd*
T0*
Tshape0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"   d      d   *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ђ
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:dd*

seed *
seed2 *
T0*
dtype0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:dd*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:dd*
T0

Variable
VariableV2*
shared_name *
	container *&
_output_shapes
:dd*
shape:dd*
dtype0
Ќ
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
R
ConstConst*
_output_shapes
:d*
valueBd*ЭЬЬ=*
dtype0
v

Variable_1
VariableV2*
shared_name *
	container *
_output_shapes
:d*
shape:d*
dtype0

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
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
Й
Conv2DConv2DReshapeVariable/read*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџdd*
use_cudnn_on_gpu(*
strides

]
addAddConv2DVariable_1/read*/
_output_shapes
:џџџџџџџџџdd*
T0
K
ReluReluadd*/
_output_shapes
:џџџџџџџџџdd*
T0
Є
MaxPoolMaxPoolRelu*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџ2d*
ksize
*
strides

q
truncated_normal_1/shapeConst*
_output_shapes
:*%
valueB"   2   d   Ш   *
dtype0
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ї
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*'
_output_shapes
:2dШ*

seed *
seed2 *
T0*
dtype0

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*'
_output_shapes
:2dШ*
T0
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*'
_output_shapes
:2dШ*
T0


Variable_2
VariableV2*
shared_name *
	container *'
_output_shapes
:2dШ*
shape:2dШ*
dtype0
Е
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:2dШ*
_class
loc:@Variable_2
x
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ
V
Const_1Const*
_output_shapes	
:Ш*
valueBШ*ЭЬЬ=*
dtype0
x

Variable_3
VariableV2*
shared_name *
	container *
_output_shapes	
:Ш*
shape:Ш*
dtype0

Variable_3/AssignAssign
Variable_3Const_1*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:Ш*
_class
loc:@Variable_3
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш
О
Conv2D_1Conv2DMaxPoolVariable_2/read*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџ2Ш*
use_cudnn_on_gpu(*
strides

b
add_1AddConv2D_1Variable_3/read*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
P
Relu_1Reluadd_1*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
Љ
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџШ*
ksize
*
strides

i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB" N  X  *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ё
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*!
_output_shapes
: и*

seed *
seed2 *
T0*
dtype0

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*!
_output_shapes
: и*
T0
v
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*!
_output_shapes
: и*
T0


Variable_4
VariableV2*
shared_name *
	container *!
_output_shapes
: и*
shape: и*
dtype0
Џ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
use_locking(*!
_output_shapes
: и*
_class
loc:@Variable_4
r
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и
V
Const_2Const*
_output_shapes	
:и*
valueBи*ЭЬЬ=*
dtype0
x

Variable_5
VariableV2*
shared_name *
	container *
_output_shapes	
:и*
shape:и*
dtype0

Variable_5/AssignAssign
Variable_5Const_2*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:и*
_class
loc:@Variable_5
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"џџџџ N  *
dtype0
r
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*)
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

MatMulMatMul	Reshape_1Variable_4/read*(
_output_shapes
:џџџџџџџџџи*
T0*
transpose_b( *
transpose_a( 
X
add_2AddMatMulVariable_5/read*(
_output_shapes
:џџџџџџџџџи*
T0
H
Relu_2Reluadd_2*(
_output_shapes
:џџџџџџџџџи*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
shape:*
dtype0
S
dropout/ShapeShapeRelu_2*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*(
_output_shapes
:џџџџџџџџџи*

seed *
seed2 *
T0*
dtype0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџи*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџи*
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
dropout/divRealDivRelu_2	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:џџџџџџџџџи*
T0
i
truncated_normal_3/shapeConst*
_output_shapes
:*
valueB"X     *
dtype0
\
truncated_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_3/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	и*

seed *
seed2 *
T0*
dtype0

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	и*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	и*
T0


Variable_6
VariableV2*
shared_name *
	container *
_output_shapes
:	и*
shape:	и*
dtype0
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	и*
_class
loc:@Variable_6
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и
T
Const_3Const*
_output_shapes
:*
valueB*ЭЬЬ=*
dtype0
v

Variable_7
VariableV2*
shared_name *
	container *
_output_shapes
:*
shape:*
dtype0

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
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:

MatMul_1MatMuldropout/mulVariable_6/read*'
_output_shapes
:џџџџџџџџџ*
T0*
transpose_b( *
transpose_a( 
Y
add_3AddMatMul_1Variable_7/read*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
J
ShapeShapeadd_3*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
L
Shape_1Shapeadd_3*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
N*
T0*

axis *
_output_shapes
:
T

Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
_output_shapes
:*
T0*

Tidx0
l
	Reshape_2Reshapeadd_3concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
k
	Reshape_3Reshapey_concat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
U
Slice_2/sizePackSub_2*
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:џџџџџџџџџ*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Q
Const_4Const*
_output_shapes
:*
valueB: *
dtype0
^
MeanMean	Reshape_4Const_4*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
`
cross_entropy/tagsConst*
_output_shapes
: *
valueB Bcross_entropy*
dtype0
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0

gradients/Mean_grad/ConstConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
valueB: *
dtype0
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Const_1Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
:*
valueB: *
dtype0
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
value	B :*
dtype0
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
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
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
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
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
_output_shapes
:*
T0
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_3_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Б
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0*
_output_shapes
:
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*(
_output_shapes
:џџџџџџџџџи*
T0*
transpose_b(*
transpose_a( 
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
_output_shapes
:	и*
T0*
transpose_b( *
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:џџџџџџџџџи
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	и
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ы
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0*
_output_shapes
:
ё
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:џџџџџџџџџи*
T0*
Tshape0
`
gradients/dropout/div_grad/NegNegRelu_2*(
_output_shapes
:џџџџџџџџџи*
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0*(
_output_shapes
:џџџџџџџџџи
ё
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
_output_shapes
:

gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*(
_output_shapes
:џџџџџџџџџи*
T0
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:и*
dtype0
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*(
_output_shapes
:џџџџџџџџџи*
T0*
Tshape0
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes	
:и*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
у
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0*(
_output_shapes
:џџџџџџџџџи
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0*
_output_shapes	
:и
Р
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*)
_output_shapes
:џџџџџџџџџ *
T0*
transpose_b(*
transpose_a( 
Д
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*!
_output_shapes
: и*
T0*
transpose_b( *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ц
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:џџџџџџџџџ 
ф
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
: и
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
_output_shapes
:*
T0
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:џџџџџџџџџШ*
T0*
Tshape0
ѕ
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџ2Ш*
ksize
*
strides


gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
g
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:Ш*
dtype0
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*0
_output_shapes
:џџџџџџџџџ2Ш*
T0*
Tshape0
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes	
:Ш*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ы
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0*0
_output_shapes
:џџџџџџџџџ2Ш
м
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes	
:Ш

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
N*
T0* 
_output_shapes
::
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(*
strides

Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(*
strides


(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:џџџџџџџџџ2d

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:2dШ
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџdd*
ksize
*
strides


gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџdd*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:d*
dtype0
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:џџџџџџџџџdd*
T0*
Tshape0
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*/
_output_shapes
:џџџџџџџџџdd
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:d

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N*
T0* 
_output_shapes
::
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(*
strides

Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(*
strides


&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:џџџџџџџџџd

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
_output_shapes
: *
valueB
 *fff?*
dtype0

beta1_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
shared_name *
	container *
shape: *
dtype0
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
_output_shapes
: *
valueB
 *wО?*
dtype0

beta2_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
shared_name *
	container *
shape: *
dtype0
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
Ё
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*&
_output_shapes
:dd*%
valueBdd*    *
dtype0
Ў
Variable/Adam
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
shared_name *
	container *
shape:dd*
dtype0
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable
{
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
Ѓ
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*&
_output_shapes
:dd*%
valueBdd*    *
dtype0
А
Variable/Adam_1
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
shared_name *
	container *
shape:dd*
dtype0
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:dd*
_class
loc:@Variable

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*&
_output_shapes
:dd

!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
_output_shapes
:d*
valueBd*    *
dtype0

Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
shared_name *
	container *
shape:d*
dtype0
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes
:d

#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
_output_shapes
:d*
valueBd*    *
dtype0

Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
shared_name *
	container *
shape:d*
dtype0
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_1
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
Ї
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*&
valueB2dШ*    *
dtype0
Д
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*
shared_name *
	container *
shape:2dШ*
dtype0
Ю
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:2dШ*
_class
loc:@Variable_2

Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ
Љ
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*&
valueB2dШ*    *
dtype0
Ж
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*
shared_name *
	container *
shape:2dШ*
dtype0
д
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:2dШ*
_class
loc:@Variable_2

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ

!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
valueBШ*    *
dtype0

Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
shared_name *
	container *
shape:Ш*
dtype0
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:Ш*
_class
loc:@Variable_3
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш

#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
valueBШ*    *
dtype0

Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
shared_name *
	container *
shape:Ш*
dtype0
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:Ш*
_class
loc:@Variable_3
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш

!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*!
_output_shapes
: и* 
valueB и*    *
dtype0
Ј
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*!
_output_shapes
: и*
shared_name *
	container *
shape: и*
dtype0
Ш
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*!
_output_shapes
: и*
_class
loc:@Variable_4
|
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и

#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*!
_output_shapes
: и* 
valueB и*    *
dtype0
Њ
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*!
_output_shapes
: и*
shared_name *
	container *
shape: и*
dtype0
Ю
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*!
_output_shapes
: и*
_class
loc:@Variable_4

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и

!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
_output_shapes	
:и*
valueBи*    *
dtype0

Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
_output_shapes	
:и*
shared_name *
	container *
shape:и*
dtype0
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:и*
_class
loc:@Variable_5
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и

#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
_output_shapes	
:и*
valueBи*    *
dtype0

Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
_output_shapes	
:и*
shared_name *
	container *
shape:и*
dtype0
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:и*
_class
loc:@Variable_5
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и

!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
_output_shapes
:	и*
valueB	и*    *
dtype0
Є
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	и*
shared_name *
	container *
shape:	и*
dtype0
Ц
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	и*
_class
loc:@Variable_6
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и

#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
_output_shapes
:	и*
valueB	и*    *
dtype0
І
Variable_6/Adam_1
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	и*
shared_name *
	container *
shape:	и*
dtype0
Ь
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	и*
_class
loc:@Variable_6
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и

!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
_output_shapes
:*
valueB*    *
dtype0

Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:

#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
_output_shapes
:*
valueB*    *
dtype0

Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *Зб8*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wО?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
к
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
T0*&
_output_shapes
:dd*
use_locking( *
_class
loc:@Variable
е
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:d*
use_locking( *
_class
loc:@Variable_1
ч
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*'
_output_shapes
:2dШ*
use_locking( *
_class
loc:@Variable_2
и
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes	
:Ш*
use_locking( *
_class
loc:@Variable_3
п
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
T0*!
_output_shapes
: и*
use_locking( *
_class
loc:@Variable_4
и
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes	
:и*
use_locking( *
_class
loc:@Variable_5
п
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:	и*
use_locking( *
_class
loc:@Variable_6
з
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:*
use_locking( *
_class
loc:@Variable_7

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
Р
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
v
ArgMaxArgMaxadd_3ArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*#
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
_
accuracyMeanCast_1Const_5*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
Z
accuracy_1/tagsConst*
_output_shapes
: *
valueB B
accuracy_1*
dtype0
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
N"gvзрZц      ЫѕVд	@ЋDё<жAJЭЬ
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514А
l
xPlaceholder*+
_output_shapes
:џџџџџџџџџd* 
shape:џџџџџџџџџd*
dtype0
e
y_Placeholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
f
Reshape/shapeConst*
_output_shapes
:*%
valueB"џџџџ   d      *
dtype0
l
ReshapeReshapexReshape/shape*/
_output_shapes
:џџџџџџџџџd*
T0*
Tshape0
o
truncated_normal/shapeConst*
_output_shapes
:*%
valueB"   d      d   *
dtype0
Z
truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ђ
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*&
_output_shapes
:dd*

seed *
dtype0*
T0*
seed2 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:dd*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:dd*
T0

Variable
VariableV2*
shared_name *
	container *
dtype0*
shape:dd*&
_output_shapes
:dd
Ќ
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*&
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
ConstConst*
_output_shapes
:d*
valueBd*ЭЬЬ=*
dtype0
v

Variable_1
VariableV2*
shared_name *
	container *
dtype0*
shape:d*
_output_shapes
:d

Variable_1/AssignAssign
Variable_1Const*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1*
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
Й
Conv2DConv2DReshapeVariable/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџdd*
strides

]
addAddConv2DVariable_1/read*/
_output_shapes
:џџџџџџџџџdd*
T0
K
ReluReluadd*/
_output_shapes
:џџџџџџџџџdd*
T0
Є
MaxPoolMaxPoolRelu*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџ2d*
ksize
*
strides

q
truncated_normal_1/shapeConst*
_output_shapes
:*%
valueB"   2   d   Ш   *
dtype0
\
truncated_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_1/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ї
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*'
_output_shapes
:2dШ*

seed *
dtype0*
T0*
seed2 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*'
_output_shapes
:2dШ*
T0
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*'
_output_shapes
:2dШ*
T0


Variable_2
VariableV2*
shared_name *
	container *
dtype0*
shape:2dШ*'
_output_shapes
:2dШ
Е
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:2dШ
x
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ
V
Const_1Const*
_output_shapes	
:Ш*
valueBШ*ЭЬЬ=*
dtype0
x

Variable_3
VariableV2*
shared_name *
	container *
dtype0*
shape:Ш*
_output_shapes	
:Ш

Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:Ш
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш
О
Conv2D_1Conv2DMaxPoolVariable_2/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџ2Ш*
strides

b
add_1AddConv2D_1Variable_3/read*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
P
Relu_1Reluadd_1*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
Љ
	MaxPool_1MaxPoolRelu_1*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџШ*
ksize
*
strides

i
truncated_normal_2/shapeConst*
_output_shapes
:*
valueB" N  X  *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0
Ё
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*!
_output_shapes
: и*

seed *
dtype0*
T0*
seed2 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*!
_output_shapes
: и*
T0
v
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*!
_output_shapes
: и*
T0


Variable_4
VariableV2*
shared_name *
	container *
dtype0*
shape: и*!
_output_shapes
: и
Џ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4*!
_output_shapes
: и
r
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и
V
Const_2Const*
_output_shapes	
:и*
valueBи*ЭЬЬ=*
dtype0
x

Variable_5
VariableV2*
shared_name *
	container *
dtype0*
shape:и*
_output_shapes	
:и

Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:и
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"џџџџ N  *
dtype0
r
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*)
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

MatMulMatMul	Reshape_1Variable_4/read*(
_output_shapes
:џџџџџџџџџи*
T0*
transpose_b( *
transpose_a( 
X
add_2AddMatMulVariable_5/read*(
_output_shapes
:џџџџџџџџџи*
T0
H
Relu_2Reluadd_2*(
_output_shapes
:џџџџџџџџџи*
T0
N
	keep_probPlaceholder*
_output_shapes
:*
shape:*
dtype0
S
dropout/ShapeShapeRelu_2*
out_type0*
_output_shapes
:*
T0
_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
_
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*(
_output_shapes
:џџџџџџџџџи*

seed *
dtype0*
T0*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
_output_shapes
: *
T0

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*(
_output_shapes
:џџџџџџџџџи*
T0

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*(
_output_shapes
:џџџџџџџџџи*
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
dropout/divRealDivRelu_2	keep_prob*
_output_shapes
:*
T0
a
dropout/mulMuldropout/divdropout/Floor*(
_output_shapes
:џџџџџџџџџи*
T0
i
truncated_normal_3/shapeConst*
_output_shapes
:*
valueB"X     *
dtype0
\
truncated_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_3/stddevConst*
_output_shapes
: *
valueB
 *ЭЬЬ=*
dtype0

"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
_output_shapes
:	и*

seed *
dtype0*
T0*
seed2 

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes
:	и*
T0
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
_output_shapes
:	и*
T0


Variable_6
VariableV2*
shared_name *
	container *
dtype0*
shape:	и*
_output_shapes
:	и
­
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	и
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и
T
Const_3Const*
_output_shapes
:*
valueB*ЭЬЬ=*
dtype0
v

Variable_7
VariableV2*
shared_name *
	container *
dtype0*
shape:*
_output_shapes
:

Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7*
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

MatMul_1MatMuldropout/mulVariable_6/read*'
_output_shapes
:џџџџџџџџџ*
T0*
transpose_b( *
transpose_a( 
Y
add_3AddMatMul_1Variable_7/read*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
J
ShapeShapeadd_3*
out_type0*
_output_shapes
:*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
L
Shape_1Shapeadd_3*
out_type0*
_output_shapes
:*
T0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
T0*
N*

axis *
_output_shapes
:
T

Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*

Tidx0*
N*
T0
l
	Reshape_2Reshapeadd_3concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
I
Shape_2Shapey_*
out_type0*
_output_shapes
:*
T0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
T0*
N*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*

Tidx0*
N*
T0
k
	Reshape_3Reshapey_concat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
U
Slice_2/sizePackSub_2*
T0*
N*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:џџџџџџџџџ*
T0*
Index0
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Q
Const_4Const*
_output_shapes
:*
valueB: *
dtype0
^
MeanMean	Reshape_4Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
cross_entropy/tagsConst*
_output_shapes
: *
valueB Bcross_entropy*
dtype0
Y
cross_entropyScalarSummarycross_entropy/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_4*
out_type0*
_output_shapes
:*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0

gradients/Mean_grad/ConstConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 

gradients/Mean_grad/Const_1Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
valueB: *
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 

gradients/Mean_grad/Maximum/yConst*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
value	B :*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
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
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
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
gradients/Reshape_2_grad/ShapeShapeadd_3*
out_type0*
_output_shapes
:*
T0
Н
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
_output_shapes
:*
T0
f
gradients/add_3_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
­
gradients/add_3_grad/SumSum gradients/Reshape_2_grad/Reshape*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Б
gradients/add_3_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
т
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
л
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0*
_output_shapes
:
С
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*(
_output_shapes
:џџџџџџџџџи*
T0*
transpose_b(*
transpose_a( 
Ж
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/add_3_grad/tuple/control_dependency*
_output_shapes
:	и*
T0*
transpose_b( *
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
э
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:џџџџџџџџџи
ъ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	и
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
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0

 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
_output_shapes
:*
T0
Н
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
І
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ы
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*
T0*
_output_shapes
:
ё
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
T0*
_output_shapes
:
f
 gradients/dropout/div_grad/ShapeShapeRelu_2*
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
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
А
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*(
_output_shapes
:џџџџџџџџџи*
T0*
Tshape0
`
gradients/dropout/div_grad/NegNegRelu_2*(
_output_shapes
:џџџџџџџџџи*
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
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
І
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ћ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape*
T0*(
_output_shapes
:џџџџџџџџџи
ё
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1*
T0*
_output_shapes
:

gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu_2*(
_output_shapes
:џџџџџџџџџи*
T0
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
_output_shapes
:*
T0
g
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:и*
dtype0
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*(
_output_shapes
:џџџџџџџџџи*
T0*
Tshape0
Џ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes	
:и*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
у
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0*(
_output_shapes
:џџџџџџџџџи
м
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0*
_output_shapes	
:и
Р
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*)
_output_shapes
:џџџџџџџџџ *
T0*
transpose_b(*
transpose_a( 
Д
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*!
_output_shapes
: и*
T0*
transpose_b( *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ц
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:џџџџџџџџџ 
ф
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
: и
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
_output_shapes
:*
T0
Ф
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*0
_output_shapes
:џџџџџџџџџШ*
T0*
Tshape0
ѕ
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
paddingSAME*
T0*0
_output_shapes
:џџџџџџџџџ2Ш*
ksize
*
strides


gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*0
_output_shapes
:џџџџџџџџџ2Ш*
T0
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
_output_shapes
:*
T0
g
gradients/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:Ш*
dtype0
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*0
_output_shapes
:џџџџџџџџџ2Ш*
T0*
Tshape0
Џ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes	
:Ш*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ы
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0*0
_output_shapes
:џџџџџџџџџ2Ш
м
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes	
:Ш

gradients/Conv2D_1_grad/ShapeNShapeNMaxPoolVariable_2/read*
out_type0*
T0*
N* 
_output_shapes
::
Ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides

Ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPool gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides


(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:џџџџџџџџџ2d

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:2dШ
ў
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
paddingSAME*
T0*/
_output_shapes
:џџџџџџџџџdd*
ksize
*
strides


gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*/
_output_shapes
:џџџџџџџџџdd*
T0
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
_output_shapes
:*
T0
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:d*
dtype0
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:џџџџџџџџџdd*
T0*
Tshape0
Љ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*/
_output_shapes
:џџџџџџџџџdd
г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:d

gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
T0*
N* 
_output_shapes
::
Ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides

Ф
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
strides


&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:џџџџџџџџџd

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:dd
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *fff?*
_output_shapes
: 

beta1_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
shared_name *
	container *
shape: *
dtype0
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*
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
beta2_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *wО?*
_output_shapes
: 

beta2_power
VariableV2*
_class
loc:@Variable*
_output_shapes
: *
shared_name *
	container *
shape: *
dtype0
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
Ё
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
Ў
Variable/Adam
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
shared_name *
	container *
shape:dd*
dtype0
Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*&
_output_shapes
:dd
{
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*&
_output_shapes
:dd
Ѓ
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueBdd*    *&
_output_shapes
:dd
А
Variable/Adam_1
VariableV2*
_class
loc:@Variable*&
_output_shapes
:dd*
shared_name *
	container *
shape:dd*
dtype0
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*&
_output_shapes
:dd

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*&
_output_shapes
:dd

!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d

Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
shared_name *
	container *
shape:d*
dtype0
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes
:d

#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueBd*    *
_output_shapes
:d

Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
_output_shapes
:d*
shared_name *
	container *
shape:d*
dtype0
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_1*
_output_shapes
:d
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes
:d
Ї
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*&
valueB2dШ*    *'
_output_shapes
:2dШ
Д
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*
shared_name *
	container *
shape:2dШ*
dtype0
Ю
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:2dШ

Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ
Љ
#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*&
valueB2dШ*    *'
_output_shapes
:2dШ
Ж
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*'
_output_shapes
:2dШ*
shared_name *
	container *
shape:2dШ*
dtype0
д
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2*'
_output_shapes
:2dШ

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*'
_output_shapes
:2dШ

!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBШ*    *
_output_shapes	
:Ш

Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
shared_name *
	container *
shape:Ш*
dtype0
Т
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:Ш
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш

#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBШ*    *
_output_shapes	
:Ш

Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
_output_shapes	
:Ш*
shared_name *
	container *
shape:Ш*
dtype0
Ш
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_3*
_output_shapes	
:Ш
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:Ш

!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0* 
valueB и*    *!
_output_shapes
: и
Ј
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*!
_output_shapes
: и*
shared_name *
	container *
shape: и*
dtype0
Ш
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4*!
_output_shapes
: и
|
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и

#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0* 
valueB и*    *!
_output_shapes
: и
Њ
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*!
_output_shapes
: и*
shared_name *
	container *
shape: и*
dtype0
Ю
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_4*!
_output_shapes
: и

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0*!
_output_shapes
: и

!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBи*    *
_output_shapes	
:и

Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
_output_shapes	
:и*
shared_name *
	container *
shape:и*
dtype0
Т
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:и
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и

#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBи*    *
_output_shapes	
:и

Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
_output_shapes	
:и*
shared_name *
	container *
shape:и*
dtype0
Ш
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_5*
_output_shapes	
:и
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes	
:и

!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	и*    *
_output_shapes
:	и
Є
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	и*
shared_name *
	container *
shape:	и*
dtype0
Ц
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	и
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и

#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueB	и*    *
_output_shapes
:	и
І
Variable_6/Adam_1
VariableV2*
_class
loc:@Variable_6*
_output_shapes
:	и*
shared_name *
	container *
shape:	и*
dtype0
Ь
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_6*
_output_shapes
:	и
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	и

!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:

Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7*
_output_shapes
:
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:

#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    *
_output_shapes
:

Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_7*
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
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *Зб8*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wО?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
к
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:dd*
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
:d*
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
:2dШ*
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
:Ш*
T0*
use_locking( *
_class
loc:@Variable_3
п
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *!
_output_shapes
: и*
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
:и*
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
:	и*
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
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(*
T0*
_class
loc:@Variable*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
validate_shape(*
T0*
_class
loc:@Variable*
_output_shapes
: 
Р
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
v
ArgMaxArgMaxadd_3ArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*#
_output_shapes
:џџџџџџџџџ*

SrcT0
*

DstT0
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
_
accuracyMeanCast_1Const_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
accuracy_1/tagsConst*
_output_shapes
: *
valueB B
accuracy_1*
dtype0
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
accuracy_1:0"З
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
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0"
train_op

Adam"Х
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
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0Ј§4       ^3\	_љ<жA*)

cross_entropyGQC


accuracy_1Ф ?&ЮMУ6       OWя	Ы4=жA*)

cross_entropyfC


accuracy_1?Y=O6       OWя	Т7=жA(*)

cross_entropy8яA


accuracy_1u?6       OWя	 W=жA<*)

cross_entropyю КA


accuracy_1#л?Ы6­6       OWя	2Аv=жAP*)

cross_entropy)эЇA


accuracy_1u?(Иr	6       OWя	&iH=жAd*)

cross_entropy7ЌA


accuracy_1О?Dчпљ6       OWя	Dн!Ж=жAx*)

cross_entropyёqA


accuracy_1э?ГжМё