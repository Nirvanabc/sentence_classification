       £K"	  @QЎЦ÷Abrain.Event:2Ћч.9≈+     n«цВ	ѕ$OQЎЦ÷A"Є„
n
xPlaceholder*
dtype0*!
shape:€€€€€€€€€ђ*,
_output_shapes
:€€€€€€€€€ђ
e
y_Placeholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
f
Reshape/shapeConst*
dtype0*%
valueB"€€€€   ,     *
_output_shapes
:
m
ReshapeReshapexReshape/shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€ђ
o
truncated_normal/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
£
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:ђd*

seed 
И
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*'
_output_shapes
:ђd
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*'
_output_shapes
:ђd
О
Variable
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
≠
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
r
Variable/readIdentityVariable*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
R
ConstConst*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_1
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Ы
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
T0*
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
Ї
Conv2DConv2DReshapeVariable/read*
strides
*
T0*
paddingVALID*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
use_cudnn_on_gpu(
|
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:d»*

seed 
О
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:d»
Р

Variable_2
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
µ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
x
Variable_2/readIdentity
Variable_2*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
V
Const_1Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
x

Variable_3
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
Ю
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
Љ
Conv2D_1Conv2DReluVariable_2/read*
strides
*
T0*
paddingVALID*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
use_cudnn_on_gpu(
Б
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:€€€€€€€€€»
®
MaxPoolMaxPoolRelu_1*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:ђd*

seed 
О
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*'
_output_shapes
:ђd
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*'
_output_shapes
:ђd
Р

Variable_4
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
µ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
x
Variable_4/readIdentity
Variable_4*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
T
Const_2Const*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_5
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Э
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
T0*
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
Њ
Conv2D_2Conv2DReshapeVariable_4/read*
strides
*
T0*
paddingVALID*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
use_cudnn_on_gpu(
А
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_3/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:d»*

seed 
О
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*'
_output_shapes
:d»
Р

Variable_6
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
µ
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
x
Variable_6/readIdentity
Variable_6*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
V
Const_3Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
x

Variable_7
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
Ю
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
Њ
Conv2D_3Conv2DRelu_2Variable_6/read*
strides
*
T0*
paddingVALID*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
use_cudnn_on_gpu(
Б
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:€€€€€€€€€»
™
	MaxPool_1MaxPoolRelu_3*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
q
truncated_normal_4/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:ђd*

seed 
О
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*'
_output_shapes
:ђd
|
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*'
_output_shapes
:ђd
Р

Variable_8
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
µ
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
x
Variable_8/readIdentity
Variable_8*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
T
Const_4Const*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_9
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Э
Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
k
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
Њ
Conv2D_4Conv2DReshapeVariable_8/read*
strides
*
T0*
paddingVALID*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
use_cudnn_on_gpu(
А
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
S
Relu_4Relu	BiasAdd_4*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_5/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *
T0*'
_output_shapes
:d»*

seed 
О
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*'
_output_shapes
:d»
С
Variable_10
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
Є
Variable_10/AssignAssignVariable_10truncated_normal_5*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
{
Variable_10/readIdentityVariable_10*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
V
Const_5Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
y
Variable_11
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
°
Variable_11/AssignAssignVariable_11Const_5*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
o
Variable_11/readIdentityVariable_11*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
њ
Conv2D_5Conv2DRelu_4Variable_10/read*
strides
*
T0*
paddingVALID*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
use_cudnn_on_gpu(
В
	BiasAdd_5BiasAddConv2D_5Variable_11/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_5Relu	BiasAdd_5*
T0*0
_output_shapes
:€€€€€€€€€»
™
	MaxPool_2MaxPoolRelu_5*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
О
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*

Tidx0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ў
`
Reshape_1/shapeConst*
dtype0*
valueB"€€€€X  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
Tshape0*
T0*(
_output_shapes
:€€€€€€€€€Ў
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
V
dropout/ShapeShape	Reshape_1*
out_type0*
T0*
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
 *  А?*
_output_shapes
: 
Э
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *
T0*(
_output_shapes
:€€€€€€€€€Ў*

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
Ц
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:€€€€€€€€€Ў
И
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:€€€€€€€€€Ў
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
:€€€€€€€€€Ў
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
 *Ќћћ=*
_output_shapes
: 
Я
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
dtype0*
seed2 *
T0*
_output_shapes
:	Ў*

seed 
Ж
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*
_output_shapes
:	Ў
t
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*
_output_shapes
:	Ў
Б
Variable_12
VariableV2*
dtype0*
	container *
shape:	Ў*
shared_name *
_output_shapes
:	Ў
∞
Variable_12/AssignAssignVariable_12truncated_normal_6*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
s
Variable_12/readIdentityVariable_12*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
T
Const_6Const*
dtype0*
valueB*Ќћћ=*
_output_shapes
:
w
Variable_13
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
†
Variable_13/AssignAssignVariable_13Const_6*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
n
Variable_13/readIdentityVariable_13*
T0*
_output_shapes
:*
_class
loc:@Variable_13
З
MatMulMatMuldropout/mulVariable_12/read*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_b( 
V
addAddMatMulVariable_13/read*
T0*'
_output_shapes
:€€€€€€€€€
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
H
ShapeShapeadd*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
J
Shape_1Shapeadd*
out_type0*
T0*
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
N*
T0*
_output_shapes
:*

axis 
T

Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
T0*
_output_shapes
:
d
concat_1/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
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
N*
T0*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
_output_shapes
:
d
concat_2/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
O
concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*

Tidx0*
N*
T0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ю
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€
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
N*
T0*
_output_shapes
:*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
_output_shapes
:€€€€€€€€€
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_7Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_7*

Tidx0*
T0*
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
 *  А?*
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
М
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
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
У
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
∆
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
Х
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
 
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( *.
_class$
" loc:@gradients/Mean_grad/Shape_1
П
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
≤
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
∞
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
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
§
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ж
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
в
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
ћ
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
a
gradients/Reshape_2_grad/ShapeShapeadd*
out_type0*
T0*
_output_shapes
:
љ
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
і
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
©
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
≠
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
Р
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Џ
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€*-
_class#
!loc:@gradients/add_grad/Reshape
”
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
Њ
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_12/read*
transpose_a( *
T0*(
_output_shapes
:€€€€€€€€€Ў*
transpose_b(
≤
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	Ў*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
е
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:€€€€€€€€€Ў*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
в
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	Ў*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
ћ
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
З
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
Ј
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
†
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
З
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
љ
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
¶
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
л
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
с
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
ћ
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Р
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
ї
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
∞
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:€€€€€€€€€Ў
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:€€€€€€€€€Ў
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
Г
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
£
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
ї
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
¶
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ы
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:€€€€€€€€€Ў*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
с
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
out_type0*
T0*
_output_shapes
:
…
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€Ў
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
gradients/concat_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
П
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
N*
out_type0*
T0*&
_output_shapes
:::
№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
к
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
М
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
л
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*.
_class$
" loc:@gradients/concat_grad/Slice
с
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*0
_class&
$"loc:@gradients/concat_grad/Slice_1
с
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*0
_class&
$"loc:@gradients/concat_grad/Slice_2
А
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradRelu_1MaxPool.gradients/concat_grad/tuple/control_dependency*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
Ж
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
Ж
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_5	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
С
gradients/Relu_1_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:€€€€€€€€€»
У
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:€€€€€€€€€»
У
gradients/Relu_5_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_5*
T0*0
_output_shapes
:€€€€€€€€€»
Р
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
ч
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
р
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
Р
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
ч
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
р
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad
Р
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradgradients/Relu_5_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_5_grad/tuple/group_depsNoOp^gradients/Relu_5_grad/ReluGrad%^gradients/BiasAdd_5_grad/BiasAddGrad
ч
1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentitygradients/Relu_5_grad/ReluGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_5_grad/ReluGrad
р
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad
Г
gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_2/read*
N*
out_type0*
T0* 
_output_shapes
::
”
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
ћ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
Н
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
К
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
Е
gradients/Conv2D_3_grad/ShapeNShapeNRelu_2Variable_6/read*
N*
out_type0*
T0* 
_output_shapes
::
”
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
ќ
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2 gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
Н
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput
К
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_5_grad/ShapeNShapeNRelu_4Variable_10/read*
N*
out_type0*
T0* 
_output_shapes
::
‘
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/read1gradients/BiasAdd_5_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
ќ
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4 gradients/Conv2D_5_grad/ShapeN:11gradients/BiasAdd_5_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
Н
(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput
К
2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter
Ъ
gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:€€€€€€€€€d
Ю
gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:€€€€€€€€€d
Ю
gradients/Relu_4_grad/ReluGradReluGrad0gradients/Conv2D_5_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:€€€€€€€€€d
Л
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
о
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
з
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
П
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
ц
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
п
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
П
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
ц
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad
п
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad
В
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
N*
out_type0*
T0* 
_output_shapes
::
Ћ
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
…
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
З
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
З
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
В
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
N*
out_type0*
T0* 
_output_shapes
::
”
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
ѕ
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
Н
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
П
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
К
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read*
N*
out_type0*
T0* 
_output_shapes
::
”
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
ѕ
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(
Н
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
П
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput
К
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *
_class
loc:@Variable
М
beta1_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
_class
loc:@Variable*
shape: 
Ђ
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
beta2_power/initial_valueConst*
dtype0*
valueB
 *wЊ?*
_output_shapes
: *
_class
loc:@Variable
М
beta2_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
_class
loc:@Variable*
shape: 
Ђ
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
£
Variable/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable
∞
Variable/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable*
shape:ђd
∆
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
|
Variable/Adam/readIdentityVariable/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
•
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable
≤
Variable/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable*
shape:ђd
ћ
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
А
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
Н
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_1
Ъ
Variable_1/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
Ѕ
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
T0*
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
П
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_1
Ь
Variable_1/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
«
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
T0*
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
І
!Variable_2/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_2
і
Variable_2/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_2*
shape:d»
ќ
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
В
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
©
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_2
ґ
Variable_2/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_2*
shape:d»
‘
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
Ж
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
П
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_3
Ь
Variable_3/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_3*
shape:»
¬
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
С
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_3
Ю
Variable_3/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_3*
shape:»
»
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
І
!Variable_4/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_4
і
Variable_4/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_4*
shape:ђd
ќ
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
В
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
©
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_4
ґ
Variable_4/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_4*
shape:ђd
‘
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
Ж
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
Н
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_5
Ъ
Variable_5/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
Ѕ
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
T0*
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
П
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_5
Ь
Variable_5/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
«
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
T0*
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
І
!Variable_6/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_6
і
Variable_6/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_6*
shape:d»
ќ
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
В
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
©
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_6
ґ
Variable_6/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_6*
shape:d»
‘
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
Ж
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
П
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_7
Ь
Variable_7/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_7*
shape:»
¬
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
С
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_7
Ю
Variable_7/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_7*
shape:»
»
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
І
!Variable_8/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_8
і
Variable_8/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_8*
shape:ђd
ќ
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
В
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
©
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_8
ґ
Variable_8/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_8*
shape:ђd
‘
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
Ж
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
Н
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_9
Ъ
Variable_9/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_9*
shape:d
Ѕ
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
П
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_9
Ь
Variable_9/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_9*
shape:d
«
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
©
"Variable_10/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_10
ґ
Variable_10/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_10*
shape:d»
“
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
Е
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
Ђ
$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_10
Є
Variable_10/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_10*
shape:d»
Ў
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
Й
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
С
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_11
Ю
Variable_11/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_11*
shape:»
∆
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
y
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
У
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_11
†
Variable_11/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_11*
shape:»
ћ
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
}
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
Щ
"Variable_12/Adam/Initializer/zerosConst*
dtype0*
valueB	Ў*    *
_output_shapes
:	Ў*
_class
loc:@Variable_12
¶
Variable_12/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	Ў*
_class
loc:@Variable_12*
shape:	Ў
 
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
}
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
Ы
$Variable_12/Adam_1/Initializer/zerosConst*
dtype0*
valueB	Ў*    *
_output_shapes
:	Ў*
_class
loc:@Variable_12
®
Variable_12/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	Ў*
_class
loc:@Variable_12*
shape:	Ў
–
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
Б
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
П
"Variable_13/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*
_class
loc:@Variable_13
Ь
Variable_13/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_13*
shape:
≈
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
x
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_13
С
$Variable_13/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*
_class
loc:@Variable_13
Ю
Variable_13/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_13*
shape:
Ћ
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
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
 *Ј—8*
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
 *wЊ?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
џ
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable
ў
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
з
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_2
№
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_3
з
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable_4
џ
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
з
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_6
№
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_7
з
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable_8
џ
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_9
м
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_10
б
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_11
в
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:	Ў*
_class
loc:@Variable_12
Џ
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *
_output_shapes
:*
_class
loc:@Variable_13
„
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
У
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
T0*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
ў

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
Ч
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
T0*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
Ц
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:€€€€€€€€€
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:€€€€€€€€€
Q
Const_8Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_8*

Tidx0*
T0*
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
: "s÷XЎОT     nµКЄ	1CgQЎЦ÷AJБ©
ћ'™'
9
Add
x"T
y"T
z"T"
Ttype:
2	
л
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ш
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
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
»
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
о
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
н
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
Р
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
”
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
л
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
2	Р
К
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
2	Р
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
К
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
2	И
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
Й
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
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514Є„
n
xPlaceholder*
dtype0*!
shape:€€€€€€€€€ђ*,
_output_shapes
:€€€€€€€€€ђ
e
y_Placeholder*
dtype0*
shape:€€€€€€€€€*'
_output_shapes
:€€€€€€€€€
f
Reshape/shapeConst*
dtype0*%
valueB"€€€€   ,     *
_output_shapes
:
m
ReshapeReshapexReshape/shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€ђ
o
truncated_normal/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
£
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:ђd
И
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*'
_output_shapes
:ђd
v
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*'
_output_shapes
:ђd
О
Variable
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
≠
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
r
Variable/readIdentityVariable*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
R
ConstConst*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_1
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Ы
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
T0*
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
Ї
Conv2DConv2DReshapeVariable/read*
strides
*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
paddingVALID
|
BiasAddBiasAddConv2DVariable_1/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
O
ReluReluBiasAdd*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:d»
О
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:d»
Р

Variable_2
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
µ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
x
Variable_2/readIdentity
Variable_2*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
V
Const_1Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
x

Variable_3
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
Ю
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
l
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
Љ
Conv2D_1Conv2DReluVariable_2/read*
strides
*
T0*
use_cudnn_on_gpu(*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
paddingVALID
Б
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_1Relu	BiasAdd_1*
T0*0
_output_shapes
:€€€€€€€€€»
®
MaxPoolMaxPoolRelu_1*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
q
truncated_normal_2/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:ђd
О
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*'
_output_shapes
:ђd
|
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*'
_output_shapes
:ђd
Р

Variable_4
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
µ
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
x
Variable_4/readIdentity
Variable_4*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
T
Const_2Const*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_5
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Э
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
T0*
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
Њ
Conv2D_2Conv2DReshapeVariable_4/read*
strides
*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
paddingVALID
А
	BiasAdd_2BiasAddConv2D_2Variable_5/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
S
Relu_2Relu	BiasAdd_2*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_3/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:d»
О
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*'
_output_shapes
:d»
Р

Variable_6
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
µ
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
x
Variable_6/readIdentity
Variable_6*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
V
Const_3Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
x

Variable_7
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
Ю
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
l
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
Њ
Conv2D_3Conv2DRelu_2Variable_6/read*
strides
*
T0*
use_cudnn_on_gpu(*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
paddingVALID
Б
	BiasAdd_3BiasAddConv2D_3Variable_7/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_3Relu	BiasAdd_3*
T0*0
_output_shapes
:€€€€€€€€€»
™
	MaxPool_1MaxPoolRelu_3*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
q
truncated_normal_4/shapeConst*
dtype0*%
valueB"   ,     d   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:ђd
О
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*'
_output_shapes
:ђd
|
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*'
_output_shapes
:ђd
Р

Variable_8
VariableV2*
dtype0*
	container *
shape:ђd*
shared_name *'
_output_shapes
:ђd
µ
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
x
Variable_8/readIdentity
Variable_8*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
T
Const_4Const*
dtype0*
valueBd*Ќћћ=*
_output_shapes
:d
v

Variable_9
VariableV2*
dtype0*
	container *
shape:d*
shared_name *
_output_shapes
:d
Э
Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
k
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
Њ
Conv2D_4Conv2DReshapeVariable_8/read*
strides
*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:€€€€€€€€€d*
data_formatNHWC*
paddingVALID
А
	BiasAdd_4BiasAddConv2D_4Variable_9/read*
data_formatNHWC*
T0*/
_output_shapes
:€€€€€€€€€d
S
Relu_4Relu	BiasAdd_4*
T0*/
_output_shapes
:€€€€€€€€€d
q
truncated_normal_5/shapeConst*
dtype0*%
valueB"      d   »   *
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
 *Ќћћ=*
_output_shapes
: 
І
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *
T0*

seed *'
_output_shapes
:d»
О
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*'
_output_shapes
:d»
|
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*'
_output_shapes
:d»
С
Variable_10
VariableV2*
dtype0*
	container *
shape:d»*
shared_name *'
_output_shapes
:d»
Є
Variable_10/AssignAssignVariable_10truncated_normal_5*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
{
Variable_10/readIdentityVariable_10*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
V
Const_5Const*
dtype0*
valueB»*Ќћћ=*
_output_shapes	
:»
y
Variable_11
VariableV2*
dtype0*
	container *
shape:»*
shared_name *
_output_shapes	
:»
°
Variable_11/AssignAssignVariable_11Const_5*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
o
Variable_11/readIdentityVariable_11*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
њ
Conv2D_5Conv2DRelu_4Variable_10/read*
strides
*
T0*
use_cudnn_on_gpu(*0
_output_shapes
:€€€€€€€€€»*
data_formatNHWC*
paddingVALID
В
	BiasAdd_5BiasAddConv2D_5Variable_11/read*
data_formatNHWC*
T0*0
_output_shapes
:€€€€€€€€€»
T
Relu_5Relu	BiasAdd_5*
T0*0
_output_shapes
:€€€€€€€€€»
™
	MaxPool_2MaxPoolRelu_5*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
M
concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
О
concatConcatV2MaxPool	MaxPool_1	MaxPool_2concat/axis*

Tidx0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ў
`
Reshape_1/shapeConst*
dtype0*
valueB"€€€€X  *
_output_shapes
:
n
	Reshape_1ReshapeconcatReshape_1/shape*
Tshape0*
T0*(
_output_shapes
:€€€€€€€€€Ў
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
V
dropout/ShapeShape	Reshape_1*
out_type0*
T0*
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
 *  А?*
_output_shapes
: 
Э
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *
T0*

seed *(
_output_shapes
:€€€€€€€€€Ў
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
Ц
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:€€€€€€€€€Ў
И
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:€€€€€€€€€Ў
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
:€€€€€€€€€Ў
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
 *Ќћћ=*
_output_shapes
: 
Я
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
dtype0*
seed2 *
T0*

seed *
_output_shapes
:	Ў
Ж
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*
_output_shapes
:	Ў
t
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*
_output_shapes
:	Ў
Б
Variable_12
VariableV2*
dtype0*
	container *
shape:	Ў*
shared_name *
_output_shapes
:	Ў
∞
Variable_12/AssignAssignVariable_12truncated_normal_6*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
s
Variable_12/readIdentityVariable_12*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
T
Const_6Const*
dtype0*
valueB*Ќћћ=*
_output_shapes
:
w
Variable_13
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
†
Variable_13/AssignAssignVariable_13Const_6*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
n
Variable_13/readIdentityVariable_13*
T0*
_output_shapes
:*
_class
loc:@Variable_13
З
MatMulMatMuldropout/mulVariable_12/read*
transpose_a( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_b( 
V
addAddMatMulVariable_13/read*
T0*'
_output_shapes
:€€€€€€€€€
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
H
ShapeShapeadd*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
J
Shape_1Shapeadd*
out_type0*
T0*
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
N*
T0*

axis *
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
Slice/size*
Index0*
T0*
_output_shapes
:
d
concat_1/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
w
concat_1ConcatV2concat_1/values_0Sliceconcat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
l
	Reshape_2Reshapeaddconcat_1*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
I
Shape_2Shapey_*
out_type0*
T0*
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
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
_output_shapes
:
d
concat_2/values_0Const*
dtype0*
valueB:
€€€€€€€€€*
_output_shapes
:
O
concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_2ConcatV2concat_2/values_0Slice_1concat_2/axis*

Tidx0*
N*
T0*
_output_shapes
:
k
	Reshape_3Reshapey_concat_2*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ю
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€
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
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
_output_shapes
:€€€€€€€€€
x
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_7Const*
dtype0*
valueB: *
_output_shapes
:
^
MeanMean	Reshape_4Const_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
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
 *  А?*
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
М
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
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_4*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
У
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
∆
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
Х
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:*.
_class$
" loc:@gradients/Mean_grad/Shape_1
 
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
П
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
≤
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
∞
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
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
{
gradients/Reshape_4_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
§
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*#
_output_shapes
:€€€€€€€€€
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ж
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
€€€€€€€€€*
_output_shapes
: 
в
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
ћ
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
a
gradients/Reshape_2_grad/ShapeShapeadd*
out_type0*
T0*
_output_shapes
:
љ
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
і
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
©
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ч
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:€€€€€€€€€
≠
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Р
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Џ
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€*-
_class#
!loc:@gradients/add_grad/Reshape
”
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
Њ
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_12/read*
transpose_a( *
T0*(
_output_shapes
:€€€€€€€€€Ў*
transpose_b(
≤
gradients/MatMul_grad/MatMul_1MatMuldropout/mul+gradients/add_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	Ў*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
е
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:€€€€€€€€€Ў*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
в
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	Ў*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
ћ
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
З
gradients/dropout/mul_grad/mulMul.gradients/MatMul_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
Ј
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
†
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
Tshape0*
T0*
_output_shapes
:
З
 gradients/dropout/mul_grad/mul_1Muldropout/div.gradients/MatMul_grad/tuple/control_dependency*
T0*
_output_shapes
:
љ
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
¶
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
л
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
с
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
i
 gradients/dropout/div_grad/ShapeShape	Reshape_1*
out_type0*
T0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
out_type0*
T0*#
_output_shapes
:€€€€€€€€€
ћ
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Р
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
ї
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
∞
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
Tshape0*
T0*(
_output_shapes
:€€€€€€€€€Ў
c
gradients/dropout/div_grad/NegNeg	Reshape_1*
T0*(
_output_shapes
:€€€€€€€€€Ў
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
Г
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
£
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
ї
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
¶
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ы
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*(
_output_shapes
:€€€€€€€€€Ў*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
с
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
d
gradients/Reshape_1_grad/ShapeShapeconcat*
out_type0*
T0*
_output_shapes
:
…
 gradients/Reshape_1_grad/ReshapeReshape3gradients/dropout/div_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:€€€€€€€€€Ў
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
gradients/concat_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
П
gradients/concat_grad/ShapeNShapeNMaxPool	MaxPool_1	MaxPool_2*
out_type0*
N*
T0*&
_output_shapes
:::
№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1gradients/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
к
gradients/concat_grad/SliceSlice gradients/Reshape_1_grad/Reshape"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р
gradients/concat_grad/Slice_1Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
р
gradients/concat_grad/Slice_2Slice gradients/Reshape_1_grad/Reshape$gradients/concat_grad/ConcatOffset:2gradients/concat_grad/ShapeN:2*
Index0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
М
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1^gradients/concat_grad/Slice_2
л
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*.
_class$
" loc:@gradients/concat_grad/Slice
с
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*0
_class&
$"loc:@gradients/concat_grad/Slice_1
с
0gradients/concat_grad/tuple/control_dependency_2Identitygradients/concat_grad/Slice_2'^gradients/concat_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*0
_class&
$"loc:@gradients/concat_grad/Slice_2
А
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradRelu_1MaxPool.gradients/concat_grad/tuple/control_dependency*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
Ж
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_3	MaxPool_10gradients/concat_grad/tuple/control_dependency_1*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
Ж
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_5	MaxPool_20gradients/concat_grad/tuple/control_dependency_2*
strides
*
T0*0
_output_shapes
:€€€€€€€€€»*
ksize
*
data_formatNHWC*
paddingVALID
С
gradients/Relu_1_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu_1*
T0*0
_output_shapes
:€€€€€€€€€»
У
gradients/Relu_3_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_3*
T0*0
_output_shapes
:€€€€€€€€€»
У
gradients/Relu_5_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_5*
T0*0
_output_shapes
:€€€€€€€€€»
Р
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad%^gradients/BiasAdd_1_grad/BiasAddGrad
ч
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
р
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
Р
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp^gradients/Relu_3_grad/ReluGrad%^gradients/BiasAdd_3_grad/BiasAddGrad
ч
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
р
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad
Р
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradgradients/Relu_5_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:»
y
)gradients/BiasAdd_5_grad/tuple/group_depsNoOp^gradients/Relu_5_grad/ReluGrad%^gradients/BiasAdd_5_grad/BiasAddGrad
ч
1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentitygradients/Relu_5_grad/ReluGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€»*1
_class'
%#loc:@gradients/Relu_5_grad/ReluGrad
р
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*
_output_shapes	
:»*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad
Г
gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_2/read*
out_type0*
N*
T0* 
_output_shapes
::
”
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
ћ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
Н
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
К
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
Е
gradients/Conv2D_3_grad/ShapeNShapeNRelu_2Variable_6/read*
out_type0*
N*
T0* 
_output_shapes
::
”
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
ќ
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2 gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
Н
(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput
К
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_5_grad/ShapeNShapeNRelu_4Variable_10/read*
out_type0*
N*
T0* 
_output_shapes
::
‘
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/read1gradients/BiasAdd_5_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
ќ
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4 gradients/Conv2D_5_grad/ShapeN:11gradients/BiasAdd_5_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
Н
(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter
О
0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput
К
2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*'
_output_shapes
:d»*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter
Ъ
gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:€€€€€€€€€d
Ю
gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:€€€€€€€€€d
Ю
gradients/Relu_4_grad/ReluGradReluGrad0gradients/Conv2D_5_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:€€€€€€€€€d
Л
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
о
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
з
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:d*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
П
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp^gradients/Relu_2_grad/ReluGrad%^gradients/BiasAdd_2_grad/BiasAddGrad
ц
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
п
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
П
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:d
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp^gradients/Relu_4_grad/ReluGrad%^gradients/BiasAdd_4_grad/BiasAddGrad
ц
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*/
_output_shapes
:€€€€€€€€€d*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad
п
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*
_output_shapes
:d*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad
В
gradients/Conv2D_grad/ShapeNShapeNReshapeVariable/read*
out_type0*
N*
T0* 
_output_shapes
::
Ћ
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
…
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
З
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
З
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
В
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_2_grad/ShapeNShapeNReshapeVariable_4/read*
out_type0*
N*
T0* 
_output_shapes
::
”
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
ѕ
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
Н
(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
П
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
К
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
Ж
gradients/Conv2D_4_grad/ShapeNShapeNReshapeVariable_8/read*
out_type0*
N*
T0* 
_output_shapes
::
”
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
ѕ
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
strides
*
T0*
use_cudnn_on_gpu(*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
paddingVALID
Н
(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter
П
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*0
_output_shapes
:€€€€€€€€€ђ*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput
К
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*'
_output_shapes
:ђd*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *
_class
loc:@Variable
М
beta1_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
_class
loc:@Variable*
shape: 
Ђ
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
beta2_power/initial_valueConst*
dtype0*
valueB
 *wЊ?*
_output_shapes
: *
_class
loc:@Variable
М
beta2_power
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
: *
_class
loc:@Variable*
shape: 
Ђ
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
£
Variable/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable
∞
Variable/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable*
shape:ђd
∆
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
|
Variable/Adam/readIdentityVariable/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
•
!Variable/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable
≤
Variable/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable*
shape:ђd
ћ
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable
А
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable
Н
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_1
Ъ
Variable_1/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
Ѕ
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
validate_shape(*
T0*
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
П
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_1
Ь
Variable_1/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_1*
shape:d
«
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
T0*
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
І
!Variable_2/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_2
і
Variable_2/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_2*
shape:d»
ќ
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
В
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
©
#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_2
ґ
Variable_2/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_2*
shape:d»
‘
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_2
Ж
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_2
П
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_3
Ь
Variable_3/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_3*
shape:»
¬
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
v
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
С
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_3
Ю
Variable_3/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_3*
shape:»
»
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_3
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_3
І
!Variable_4/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_4
і
Variable_4/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_4*
shape:ђd
ќ
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
В
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
©
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_4
ґ
Variable_4/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_4*
shape:ђd
‘
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_4
Ж
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_4
Н
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_5
Ъ
Variable_5/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
Ѕ
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*
T0*
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
П
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_5
Ь
Variable_5/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_5*
shape:d
«
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*
T0*
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
І
!Variable_6/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_6
і
Variable_6/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_6*
shape:d»
ќ
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
В
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
©
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_6
ґ
Variable_6/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_6*
shape:d»
‘
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_6
Ж
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_6
П
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_7
Ь
Variable_7/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_7*
shape:»
¬
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
v
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
С
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_7
Ю
Variable_7/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_7*
shape:»
»
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_7
z
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_7
І
!Variable_8/Adam/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_8
і
Variable_8/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_8*
shape:ђd
ќ
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
В
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
©
#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*&
valueBђd*    *'
_output_shapes
:ђd*
_class
loc:@Variable_8
ґ
Variable_8/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:ђd*
_class
loc:@Variable_8*
shape:ђd
‘
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:ђd*
_class
loc:@Variable_8
Ж
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*'
_output_shapes
:ђd*
_class
loc:@Variable_8
Н
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_9
Ъ
Variable_9/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_9*
shape:d
Ѕ
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
П
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
valueBd*    *
_output_shapes
:d*
_class
loc:@Variable_9
Ь
Variable_9/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:d*
_class
loc:@Variable_9*
shape:d
«
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:d*
_class
loc:@Variable_9
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_output_shapes
:d*
_class
loc:@Variable_9
©
"Variable_10/Adam/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_10
ґ
Variable_10/Adam
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_10*
shape:d»
“
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
Е
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
Ђ
$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*&
valueBd»*    *'
_output_shapes
:d»*
_class
loc:@Variable_10
Є
Variable_10/Adam_1
VariableV2*
dtype0*
	container *
shared_name *'
_output_shapes
:d»*
_class
loc:@Variable_10*
shape:d»
Ў
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:d»*
_class
loc:@Variable_10
Й
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*'
_output_shapes
:d»*
_class
loc:@Variable_10
С
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_11
Ю
Variable_11/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_11*
shape:»
∆
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
y
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
У
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
valueB»*    *
_output_shapes	
:»*
_class
loc:@Variable_11
†
Variable_11/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:»*
_class
loc:@Variable_11*
shape:»
ћ
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:»*
_class
loc:@Variable_11
}
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_output_shapes	
:»*
_class
loc:@Variable_11
Щ
"Variable_12/Adam/Initializer/zerosConst*
dtype0*
valueB	Ў*    *
_output_shapes
:	Ў*
_class
loc:@Variable_12
¶
Variable_12/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	Ў*
_class
loc:@Variable_12*
shape:	Ў
 
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
}
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
Ы
$Variable_12/Adam_1/Initializer/zerosConst*
dtype0*
valueB	Ў*    *
_output_shapes
:	Ў*
_class
loc:@Variable_12
®
Variable_12/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	Ў*
_class
loc:@Variable_12*
shape:	Ў
–
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	Ў*
_class
loc:@Variable_12
Б
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_output_shapes
:	Ў*
_class
loc:@Variable_12
П
"Variable_13/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*
_class
loc:@Variable_13
Ь
Variable_13/Adam
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_13*
shape:
≈
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
x
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_13
С
$Variable_13/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*
_class
loc:@Variable_13
Ю
Variable_13/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:*
_class
loc:@Variable_13*
shape:
Ћ
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@Variable_13
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
 *Ј—8*
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
 *wЊ?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
џ
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable
ў
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_1
з
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_2
№
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_3
з
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable_4
џ
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_5
з
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_6
№
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_7
з
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:ђd*
_class
loc:@Variable_8
џ
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes
:d*
_class
loc:@Variable_9
м
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *'
_output_shapes
:d»*
_class
loc:@Variable_10
б
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes	
:»*
_class
loc:@Variable_11
в
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes
:	Ў*
_class
loc:@Variable_12
Џ
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
use_locking( *
_output_shapes
:*
_class
loc:@Variable_13
„
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
У
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
T0*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
ў

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable
Ч
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
T0*
use_locking( *
_output_shapes
: *
_class
loc:@Variable
Ц
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam^Adam/Assign^Adam/Assign_1
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
w
ArgMax_1ArgMaxy_ArgMax_1/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:€€€€€€€€€
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:€€€€€€€€€
Q
Const_8Const*
dtype0*
valueB: *
_output_shapes
:
_
accuracyMeanCast_1Const_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
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
: ""ш
trainable_variablesаЁ
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
Variable_13:0Variable_13/AssignVariable_13/read:02	Const_6:0"≤!
	variables§!°!
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
Adam".
	summaries!

cross_entropy:0
accuracy_1:0ж4Є4       ^3\	ёRЎЦ÷A*)

cross_entropy±oЂ?


accuracy_1бz?uZ>ч6       OWМп	4№SЎЦ÷A2*)

cross_entropyгж{?


accuracy_1Є?іhы'6       OWМп	t,нSЎЦ÷Ad*)

cross_entropytl?


accuracy_1RЄ?яЭЩр7       зи Y	Ў,ЊTЎЦ÷AЦ*)

cross_entropy1ЄЋ?


accuracy_1RЄ?“uµF7       зи Y	Л©UЎЦ÷A»*)

cross_entropyJF@


accuracy_1Ел>2Й Э7       зи Y	qLvVЎЦ÷Aъ*)

cross_entropy§£b?


accuracy_1П¬х>uћп7       зи Y	Q—CWЎЦ÷Aђ*)

cross_entropyJhw?


accuracy_1П¬х>.H§