       ŁK"	  Ŕ%ÖAbrain.Event:2ÖUÝęC     ĄýŐ<	é%ÖA"Ý
W
inputsPlaceholder*
_output_shapes

:dd*
shape
:dd*
dtype0
X
targetsPlaceholder*
_output_shapes

:dd*
shape
:dd*
dtype0
N
	keep_probPlaceholder*
_output_shapes
:*
shape:*
dtype0
]
DropoutWrapperInit/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
_
DropoutWrapperInit_1/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
a
DropoutWrapperInit_1/Const_1Const*
valueB
 *  ?*
_output_shapes
: *
dtype0

JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:d*
_output_shapes
:*
dtype0

PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
ý
KMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:*

Tidx0

PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0

JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillKMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatPMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:d*
_output_shapes
:*
dtype0

RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

MMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0

RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
 
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillMMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:d*
_output_shapes
:*
dtype0

LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
valueB:d*
_output_shapes
:*
dtype0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
valueB:d*
_output_shapes
:*
dtype0

RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0

MMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:*

Tidx0

RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
 
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFillMMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatRMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
valueB:d*
_output_shapes
:*
dtype0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
valueB:d*
_output_shapes
:*
dtype0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
valueB:d*
_output_shapes
:*
dtype0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
valueB:d*
_output_shapes
:*
dtype0

TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

OMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV2NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0

TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Ś
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1FillOMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
valueB:d*
_output_shapes
:*
dtype0

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
valueB:d*
_output_shapes
:*
dtype0
U
one_hot/on_valueConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
V
one_hot/off_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
P
one_hot/depthConst*
value
B :*
_output_shapes
: *
dtype0

one_hotOneHotinputsone_hot/depthone_hot/on_valueone_hot/off_value*
TI0*#
_output_shapes
:dd*
axis˙˙˙˙˙˙˙˙˙*
T0
F
RankConst*
value	B :*
_output_shapes
: *
dtype0
M
range/startConst*
value	B :*
_output_shapes
: *
dtype0
M
range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
V
rangeRangerange/startRankrange/delta*
_output_shapes
:*

Tidx0
`
concat/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
M
concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
q
concatConcatV2concat/values_0rangeconcat/axis*
N*
T0*
_output_shapes
:*

Tidx0
b
	transpose	Transposeone_hotconcat*#
_output_shapes
:dd*
Tperm0*
T0
^
	rnn/ShapeConst*!
valueB"d   d      *
_output_shapes
:*
dtype0
a
rnn/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
c
rnn/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
c
rnn/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
end_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
S
	rnn/ConstConst*
valueB:d*
_output_shapes
:*
dtype0
U
rnn/Const_1Const*
valueB:d*
_output_shapes
:*
dtype0
Q
rnn/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
y

rnn/concatConcatV2	rnn/Constrnn/Const_1rnn/concat/axis*
N*
T0*
_output_shapes
:*

Tidx0
T
rnn/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
W
	rnn/zerosFill
rnn/concatrnn/zeros/Const*
_output_shapes

:dd*
T0
J
rnn/timeConst*
value	B : *
_output_shapes
: *
dtype0
Ň
rnn/TensorArrayTensorArrayV3rnn/strided_slice*/
tensor_array_namernn/dynamic_rnn/output_0*
element_shape:*
clear_after_read(*
dtype0*
dynamic_size( *
_output_shapes

:: 
Ó
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*.
tensor_array_namernn/dynamic_rnn/input_0*
element_shape:*
clear_after_read(*
dtype0*
dynamic_size( *
_output_shapes

:: 
q
rnn/TensorArrayUnstack/ShapeConst*!
valueB"d   d      *
_output_shapes
:*
dtype0
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
ě
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
end_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
d
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Ä
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ć
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
T0*
_output_shapes
: *
_class
loc:@transpose

rnn/while/EnterEnterrnn/time*
parallel_iterations *
is_constant( *
_output_shapes
: *'

frame_namernn/while/while_context*
T0
Ľ
rnn/while/Enter_1Enterrnn/TensorArray:1*
parallel_iterations *
is_constant( *
_output_shapes
: *'

frame_namernn/while/while_context*
T0
ć
rnn/while/Enter_2EnterJMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
is_constant( *
_output_shapes

:dd*'

frame_namernn/while/while_context*
T0
č
rnn/while/Enter_3EnterLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
is_constant( *
_output_shapes

:dd*'

frame_namernn/while/while_context*
T0
č
rnn/while/Enter_4EnterLMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*
parallel_iterations *
is_constant( *
_output_shapes

:dd*'

frame_namernn/while/while_context*
T0
ę
rnn/while/Enter_5EnterNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
is_constant( *
_output_shapes

:dd*'

frame_namernn/while/while_context*
T0
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
_output_shapes
: : *
N
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
_output_shapes
: : *
N
|
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0* 
_output_shapes
:dd: *
N
|
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0* 
_output_shapes
:dd: *
N
|
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0* 
_output_shapes
:dd: *
N
|
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
T0* 
_output_shapes
:dd: *
N
¨
rnn/while/Less/EnterEnterrnn/strided_slice*
parallel_iterations *
is_constant(*
_output_shapes
: *'

frame_namernn/while/while_context*
T0
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
_output_shapes
: *
T0
F
rnn/while/LoopCondLoopCondrnn/while/Less*
_output_shapes
: 

rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
T0*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge

rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1

rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_2

rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_3

rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*
T0*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_4

rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*
T0*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_5
S
rnn/while/IdentityIdentityrnn/while/Switch:1*
_output_shapes
: *
T0
W
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
_output_shapes
: *
T0
_
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
_output_shapes

:dd*
T0
_
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
_output_shapes

:dd*
T0
_
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
_output_shapes

:dd*
T0
_
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
_output_shapes

:dd*
T0
š
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ä
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant(*
_output_shapes
: *'

frame_namernn/while/while_context*
T0
ş
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
_output_shapes
:	d*
dtype0
ç
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"     *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ÝĂ˝*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ÝĂ=*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ó
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
dtype0*

seed *
T0*
seed2 *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ţ
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: *
T0
ň
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
T0
ä
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
T0
í
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shape:
*
dtype0*
shared_name *
	container * 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
Ů
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel

5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
T0
Ň
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ß
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ă
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:*
T0

Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0

Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
N*
T0*
_output_shapes
:	d*

Tidx0

Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(* 
_output_shapes
:
*'

frame_namernn/while/while_context*
T0
§
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
transpose_b( *
transpose_a( *
_output_shapes
:	d*
T0

Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
parallel_iterations *
is_constant(*
_output_shapes	
:*'

frame_namernn/while/while_context*
T0

Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d*
T0

@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ą
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
ą
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0

@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
ô
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
ś
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
_output_shapes

:dd*
T0
Č
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMulrnn/while/Identity_2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
ş
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
_output_shapes

:dd*
T0
´
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
÷
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
ň
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
´
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
ź
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
ů
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0

1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^rnn/while/Identity*
valueB"d   d   *
_output_shapes
:*
dtype0

>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^rnn/while/Identity*
valueB
 *    *
_output_shapes
: *
dtype0

>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ű
Hrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*

seed *
dtype0*
_output_shapes

:dd*
seed2 *
T0
ć
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
: *
T0
ř
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
ę
:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes

:dd*
T0
Ă
5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/EnterEnter	keep_prob*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ě
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
_output_shapes
:*
T0

1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
_output_shapes
:*
T0
Ö
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
Ă
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
_output_shapes

:dd*
T0
ç
Qrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"Č     *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
Ů
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ˝*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
Ů
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
Ó
Yrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Č*
dtype0*

seed *
T0*
seed2 *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
Ţ
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
_output_shapes
: *
T0
ň
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
Č*
T0
ä
Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
Č*
T0
í
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
VariableV2*
shape:
Č*
dtype0*
shared_name *
	container * 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
Ů
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel

5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
Č*
T0
Ň
@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
ß
.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
Ă
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias

3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:*
T0

Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0

Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatConcatV2/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulrnn/while/Identity_5Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axis*
N*
T0*
_output_shapes
:	dČ*

Tidx0

Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(* 
_output_shapes
:
Č*'

frame_namernn/while/while_context*
T0
§
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter*
transpose_b( *
transpose_a( *
_output_shapes
:	d*
T0

Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read*
parallel_iterations *
is_constant(*
_output_shapes	
:*'

frame_namernn/while/while_context*
T0

Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d*
T0

@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Ą
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
ą
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0

@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
ô
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
ś
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add*
_output_shapes

:dd*
T0
Č
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mulMulrnn/while/Identity_4Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
ş
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split*
_output_shapes

:dd*
T0
´
?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
÷
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
ň
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
´
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
ź
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
ů
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0

1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/ShapeConst^rnn/while/Identity*
valueB"d   d   *
_output_shapes
:*
dtype0

>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/minConst^rnn/while/Identity*
valueB
 *    *
_output_shapes
: *
dtype0

>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/maxConst^rnn/while/Identity*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ű
Hrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Shape*

seed *
dtype0*
_output_shapes

:dd*
seed2 *
T0
ć
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes
: *
T0
ř
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
ę
:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes

:dd*
T0
Ě
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform*
_output_shapes
:*
T0

1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/add*
_output_shapes
:*
T0
Ö
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
Ă
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
_output_shapes

:dd*
T0

3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
parallel_iterations *
_output_shapes
:*
T0*
is_constant(*'

frame_namernn/while/while_context*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul
¸
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulrnn/while/Identity_1*
T0*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul
f
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
_output_shapes
: *
T0
X
rnn/while/NextIterationNextIterationrnn/while/add*
_output_shapes
: *
T0
z
rnn/while/NextIteration_1NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

rnn/while/NextIteration_2NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0

rnn/while/NextIteration_3NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:dd*
T0

rnn/while/NextIteration_4NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0

rnn/while/NextIteration_5NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
_output_shapes

:dd*
T0
I
rnn/while/ExitExitrnn/while/Switch*
_output_shapes
: *
T0
M
rnn/while/Exit_1Exitrnn/while/Switch_1*
_output_shapes
: *
T0
U
rnn/while/Exit_2Exitrnn/while/Switch_2*
_output_shapes

:dd*
T0
U
rnn/while/Exit_3Exitrnn/while/Switch_3*
_output_shapes

:dd*
T0
U
rnn/while/Exit_4Exitrnn/while/Switch_4*
_output_shapes

:dd*
T0
U
rnn/while/Exit_5Exitrnn/while/Switch_5*
_output_shapes

:dd*
T0

&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/startConst*"
_class
loc:@rnn/TensorArray*
_output_shapes
: *
value	B : *
dtype0

 rnn/TensorArrayStack/range/deltaConst*"
_class
loc:@rnn/TensorArray*
_output_shapes
: *
value	B :*
dtype0
ä
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@rnn/TensorArray
đ
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
dtype0*
element_shape
:dd*"
_output_shapes
:ddd*"
_class
loc:@rnn/TensorArray
\
rnn/Const_2Const*
valueB"d   d   *
_output_shapes
:*
dtype0
U
rnn/Const_3Const*
valueB:d*
_output_shapes
:*
dtype0
J
rnn/RankConst*
value	B :*
_output_shapes
: *
dtype0
Q
rnn/range/startConst*
value	B :*
_output_shapes
: *
dtype0
Q
rnn/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*
_output_shapes
:*

Tidx0
f
rnn/concat_1/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
S
rnn/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0

rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0

rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*"
_output_shapes
:ddd*
Tperm0*
T0
U
concat_1/concat_dimConst*
value	B :*
_output_shapes
: *
dtype0
P
concat_1Identityrnn/transpose*"
_output_shapes
:ddd*
T0
^
Reshape/shapeConst*
valueB"˙˙˙˙d   *
_output_shapes
:*
dtype0
c
ReshapeReshapeconcat_1Reshape/shape*
Tshape0*
_output_shapes
:	Nd*
T0
o
softmax/truncated_normal/shapeConst*
valueB"d      *
_output_shapes
:*
dtype0
b
softmax/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
d
softmax/truncated_normal/stddevConst*
valueB
 *ÍĚĚ=*
_output_shapes
: *
dtype0
Ť
(softmax/truncated_normal/TruncatedNormalTruncatedNormalsoftmax/truncated_normal/shape*

seed *
dtype0*
_output_shapes
:	d*
seed2 *
T0

softmax/truncated_normal/mulMul(softmax/truncated_normal/TruncatedNormalsoftmax/truncated_normal/stddev*
_output_shapes
:	d*
T0

softmax/truncated_normalAddsoftmax/truncated_normal/mulsoftmax/truncated_normal/mean*
_output_shapes
:	d*
T0

softmax/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
:	d*
shape:	d*
	container 
Ĺ
softmax/Variable/AssignAssignsoftmax/Variablesoftmax/truncated_normal*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable

softmax/Variable/readIdentitysoftmax/Variable*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
\
softmax/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0

softmax/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
shape:*
	container 
ź
softmax/Variable_1/AssignAssignsoftmax/Variable_1softmax/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

softmax/Variable_1/readIdentitysoftmax/Variable_1*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

MatMulMatMulReshapesoftmax/Variable/read*
transpose_b( *
transpose_a( * 
_output_shapes
:
N*
T0
V
addAddMatMulsoftmax/Variable_1/read* 
_output_shapes
:
N*
T0
F
predictionsSoftmaxadd* 
_output_shapes
:
N*
T0
W
one_hot_1/on_valueConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
X
one_hot_1/off_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
R
one_hot_1/depthConst*
value
B :*
_output_shapes
: *
dtype0
Ł
	one_hot_1OneHottargetsone_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
TI0*#
_output_shapes
:dd*
axis˙˙˙˙˙˙˙˙˙*
T0
`
Reshape_1/shapeConst*
valueB"'     *
_output_shapes
:*
dtype0
i
	Reshape_1Reshape	one_hot_1Reshape_1/shape*
Tshape0* 
_output_shapes
:
N*
T0
H
Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
V
ShapeConst*
valueB"'     *
_output_shapes
:*
dtype0
H
Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
X
Shape_1Const*
valueB"'     *
_output_shapes
:*
dtype0
G
Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
:
SubSubRank_2Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
N*

axis *
_output_shapes
:*
T0
T

Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
_output_shapes
:*
T0
d
concat_2/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
O
concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
w
concat_2ConcatV2concat_2/values_0Sliceconcat_2/axis*
N*
T0*
_output_shapes
:*

Tidx0
\
	Reshape_2Reshapeaddconcat_2*
Tshape0* 
_output_shapes
:
N*
T0
H
Rank_3Const*
value	B :*
_output_shapes
: *
dtype0
X
Shape_2Const*
valueB"'     *
_output_shapes
:*
dtype0
I
Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
>
Sub_1SubRank_3Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
N*

axis *
_output_shapes
:*
T0
V
Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
_output_shapes
:*
T0
d
concat_3/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
dtype0
O
concat_3/axisConst*
value	B : *
_output_shapes
: *
dtype0
y
concat_3ConcatV2concat_3/values_0Slice_1concat_3/axis*
N*
T0*
_output_shapes
:*

Tidx0
b
	Reshape_3Reshape	Reshape_1concat_3*
Tshape0* 
_output_shapes
:
N*
T0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*'
_output_shapes
:N:
N*
T0
I
Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
>
Sub_2SubRank_1Sub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
U
Slice_2/sizePackSub_2*
N*

axis *
_output_shapes
:*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
p
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
_output_shapes	
:N*
T0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
\
MeanMean	Reshape_4Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Z
batch_loss/tagsConst*
valueB B
batch_loss*
_output_shapes
: *
dtype0
S

batch_lossScalarSummarybatch_loss/tagsMean*
_output_shapes
: *
T0
O
Merge/MergeSummaryMergeSummary
batch_loss*
_output_shapes
: *
N
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
T
gradients/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
S
gradients/f_countConst*
value	B : *
_output_shapes
: *
dtype0
§
gradients/f_count_1Entergradients/f_count*
parallel_iterations *
is_constant( *
_output_shapes
: *'

frame_namernn/while/while_context*
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
_output_shapes
: : *
N
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
_output_shapes
: : *
T0
f
gradients/Add/yConst^rnn/while/Identity*
value	B :*
_output_shapes
: *
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
á
gradients/NextIterationNextIterationgradients/Add[^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2^^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2f^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2^^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2f^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2*
_output_shapes
: *
T0
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
S
gradients/b_countConst*
value	B :*
_output_shapes
: *
dtype0
ł
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
is_constant( *
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context*
T0
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
_output_shapes
: : *
N
ş
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *
is_constant(*
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context*
T0
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
˛
gradients/NextIteration_1NextIterationgradients/SubV^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_3Exitgradients/Switch_1*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
m
"gradients/Mean_grad/Tile/multiplesConst*
valueB:N*
_output_shapes
:*
dtype0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:N*
T0
d
gradients/Mean_grad/ShapeConst*
valueB:N*
_output_shapes
:*
dtype0
^
gradients/Mean_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0

gradients/Mean_grad/ConstConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
Â
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape

gradients/Mean_grad/Const_1Const*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:*
valueB: *
dtype0
Č
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( *,
_class"
 loc:@gradients/Mean_grad/Shape

gradients/Mean_grad/Maximum/yConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
°
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape
Ž
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_output_shapes	
:N*
T0
i
gradients/Reshape_4_grad/ShapeConst*
valueB:N*
_output_shapes
:*
dtype0

 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
_output_shapes	
:N*
T0
m
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
N*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0
Ú
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
_output_shapes
:	N*
T0
ź
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
N*
T0
o
gradients/Reshape_2_grad/ShapeConst*
valueB"'     *
_output_shapes
:*
dtype0
ś
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0* 
_output_shapes
:
N*
T0
i
gradients/add_grad/ShapeConst*
valueB"'     *
_output_shapes
:*
dtype0
e
gradients/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Š
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0* 
_output_shapes
:
N*
T0
­
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0
Š
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/Reshapesoftmax/Variable/read*
transpose_b(*
transpose_a( *
_output_shapes
:	Nd*
T0

gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_b( *
transpose_a(*
_output_shapes
:	d*
T0
q
gradients/Reshape_grad/ShapeConst*!
valueB"d   d   d   *
_output_shapes
:*
dtype0
 
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*"
_output_shapes
:ddd*
T0
v
.gradients/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/concat_1*
_output_shapes
:*
T0
˝
&gradients/rnn/transpose_grad/transpose	Transposegradients/Reshape_grad/Reshape.gradients/rnn/transpose_grad/InvertPermutation*"
_output_shapes
:ddd*
Tperm0*
T0
ę
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_1*
source	gradients*
_output_shapes

:: *"
_class
loc:@rnn/TensorArray

Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_1Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range&gradients/rnn/transpose_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
d
gradients/zerosConst*
valueBdd*    *
_output_shapes

:dd*
dtype0
f
gradients/zeros_1Const*
valueBdd*    *
_output_shapes

:dd*
dtype0
f
gradients/zeros_2Const*
valueBdd*    *
_output_shapes

:dd*
dtype0
f
gradients/zeros_3Const*
valueBdd*    *
_output_shapes

:dd*
dtype0

&gradients/rnn/while/Exit_1_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant( *
_output_shapes
: *1

frame_name#!gradients/rnn/while/while_context*
T0
Ę
&gradients/rnn/while/Exit_2_grad/b_exitEntergradients/zeros*
parallel_iterations *
is_constant( *
_output_shapes

:dd*1

frame_name#!gradients/rnn/while/while_context*
T0
Ě
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_1*
parallel_iterations *
is_constant( *
_output_shapes

:dd*1

frame_name#!gradients/rnn/while/while_context*
T0
Ě
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_2*
parallel_iterations *
is_constant( *
_output_shapes

:dd*1

frame_name#!gradients/rnn/while/while_context*
T0
Ě
&gradients/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_3*
parallel_iterations *
is_constant( *
_output_shapes

:dd*1

frame_name#!gradients/rnn/while/while_context*
T0
ş
*gradients/rnn/while/Switch_1_grad/b_switchMerge&gradients/rnn/while/Exit_1_grad/b_exit1gradients/rnn/while/Switch_1_grad_1/NextIteration*
T0*
_output_shapes
: : *
N
Â
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration*
T0* 
_output_shapes
:dd: *
N
Â
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration*
T0* 
_output_shapes
:dd: *
N
Â
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration*
T0* 
_output_shapes
:dd: *
N
Â
*gradients/rnn/while/Switch_5_grad/b_switchMerge&gradients/rnn/while/Exit_5_grad/b_exit1gradients/rnn/while/Switch_5_grad_1/NextIteration*
T0* 
_output_shapes
:dd: *
N
Ô
'gradients/rnn/while/Merge_1_grad/SwitchSwitch*gradients/rnn/while/Switch_1_grad/b_switchgradients/b_count_2*
T0*
_output_shapes
: : *=
_class3
1/loc:@gradients/rnn/while/Switch_1_grad/b_switch
ä
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
ä
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
ä
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
ä
'gradients/rnn/while/Merge_5_grad/SwitchSwitch*gradients/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*
T0*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
w
%gradients/rnn/while/Enter_1_grad/ExitExit'gradients/rnn/while/Merge_1_grad/Switch*
_output_shapes
: *
T0

%gradients/rnn/while/Enter_2_grad/ExitExit'gradients/rnn/while/Merge_2_grad/Switch*
_output_shapes

:dd*
T0

%gradients/rnn/while/Enter_3_grad/ExitExit'gradients/rnn/while/Merge_3_grad/Switch*
_output_shapes

:dd*
T0

%gradients/rnn/while/Enter_4_grad/ExitExit'gradients/rnn/while/Merge_4_grad/Switch*
_output_shapes

:dd*
T0

%gradients/rnn/while/Enter_5_grad/ExitExit'gradients/rnn/while/Merge_5_grad/Switch*
_output_shapes

:dd*
T0
Č
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
parallel_iterations *
_output_shapes
:*
T0*
is_constant(*1

frame_name#!gradients/rnn/while/while_context*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul
ý
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter)gradients/rnn/while/Merge_1_grad/Switch:1*
source	gradients*
_output_shapes

:: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul
×
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity)gradients/rnn/while/Merge_1_grad/Switch:1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul
Ď
]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*%
_class
loc:@rnn/while/Identity*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
¤
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *%
_class
loc:@rnn/while/Identity
Ż
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0

Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity^gradients/Add*
swap_memory( *
_output_shapes
: *
T0
Ä
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
Ő
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
§
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes

:dd*
dtype0
ź
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0
Ŕ
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0

cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
â
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
â
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
č
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*
	elem_type0*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1
ż
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
č
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
â
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_sizeConst*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ë
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulMulNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
Ł
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
_output_shapes
:*
T0
â
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_sizeConst*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
­
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
˘
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ď
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes
:*
T0
Š
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
Tshape0*
_output_shapes
:*
T0
Ľ
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
Ů
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0

cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ä
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ä
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ä
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/EnterEnter	keep_prob*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
§
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
ń
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_sizeConst*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ş
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ľ
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ń
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Á
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
ţ
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
_output_shapes
:*
T0

1gradients/rnn/while/Switch_1_grad_1/NextIterationNextIteration)gradients/rnn/while/Merge_1_grad/Switch:1*
_output_shapes
: *
T0
ú
gradients/AddNAddN)gradients/rnn/while/Merge_5_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape*
N*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ŕ
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ű
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ó
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
á
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ü
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
÷
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN*
_output_shapes

:dd*
T0
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
É
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0

gradients/AddN_1AddN)gradients/rnn/while/Merge_4_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGrad*
N*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ú
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid
ľ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ő
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ę
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
¸
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Đ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
Ö
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*'
_class
loc:@rnn/while/Identity_4*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
°
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *'
_class
loc:@rnn/while/Identity_4
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ť
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_4^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ű
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ö
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ž
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ä
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ß
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Â
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
Ě
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
Ç
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
Â
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¨
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
_output_shapes
: *
dtype0
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ü
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
ŕ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˝
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Ž
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
ń
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/Const*
N*
T0*
_output_shapes
:	d*

Tidx0

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
Š
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(* 
_output_shapes
:
Č*1

frame_name#!gradients/rnn/while/while_context*
T0
č
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *
_output_shapes
:	dČ*
T0

hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
é
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat
Ĺ
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ĺ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat^gradients/Add*
swap_memory( *
_output_shapes
:	dČ*
T0
Ú
jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
 
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	dČ
ň
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
transpose_b( *
transpose_a(* 
_output_shapes
:
Č*
T0
Ź
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
_output_shapes	
:*
dtype0
Î
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes	
:*1

frame_name#!gradients/rnn/while/while_context*
T0
á
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
_output_shapes
	:: *
N

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
É
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ů
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
í
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
§
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
Ź
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
´
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
ˇ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
š
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
 
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
Ľ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape*
Index0*
_output_shapes

:dd*
T0
Ť
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1*
Index0*
_output_shapes

:dd*
T0
ľ
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Č*    * 
_output_shapes
:
Č*
dtype0
Ń
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
is_constant( * 
_output_shapes
:
Č*1

frame_name#!gradients/rnn/while/while_context*
T0
ă
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*"
_output_shapes
:
Č: *
N

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
Č:
Č*
T0
Č
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
Č*
T0
ü
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
Č*
T0
đ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
Č*
T0
ź
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0
Ŕ
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0

cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
â
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
â
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
č
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*
	elem_type0*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1
ż
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
č
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
â
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ë
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
Ł
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
_output_shapes
:*
T0
â
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
­
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
˘
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^gradients/Add*
swap_memory( *
_output_shapes
:*
T0
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ď
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice*
_output_shapes
:*
T0
Š
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
Tshape0*
_output_shapes
:*
T0
Ľ
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
Ů
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0*
T0

cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ä
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ä
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1^gradients/Add*
swap_memory( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
§
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
ń
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ş
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ľ
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
ń
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Á
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
ţ
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
§
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
_output_shapes
:*
T0
Ĺ
1gradients/rnn/while/Switch_5_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0
ü
gradients/AddN_2AddN)gradients/rnn/while/Merge_3_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
N*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ŕ
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ű
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ő
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN_2^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
á
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ü
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ů
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN_2*
_output_shapes

:dd*
T0
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
É
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0

gradients/AddN_3AddN)gradients/rnn/while/Merge_2_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
N*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_3egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_3ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ú
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid
ľ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ő
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ę
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
¸
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Đ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
Ö
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*'
_class
loc:@rnn/while/Identity_2*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
°
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *'
_class
loc:@rnn/while/Identity_2
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ť
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_2^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ű
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
Ö
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ž
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0

dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ä
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ß
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^gradients/Add*
swap_memory( *
_output_shapes

:dd*
T0
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Â
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
Tshape0*
_output_shapes

:dd*
T0
Ě
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
Ç
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
Â
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
¨
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
_output_shapes
: *
dtype0
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ü
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
Tshape0*
_output_shapes

:dd*
T0
ŕ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˝
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
Ž
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
ń
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*
N*
T0*
_output_shapes
:	d*

Tidx0

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
Š
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(* 
_output_shapes
:
*1

frame_name#!gradients/rnn/while/while_context*
T0
č
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *
_output_shapes
:	d*
T0

hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
é
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*
	elem_type0*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat
Ĺ
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*'

frame_namernn/while/while_context*
T0
ĺ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^gradients/Add*
swap_memory( *
_output_shapes
:	d*
T0
Ú
jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
_output_shapes
:*1

frame_name#!gradients/rnn/while/while_context*
T0
 
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	d
ň
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
transpose_b( *
transpose_a(* 
_output_shapes
:
*
T0
Ź
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
_output_shapes	
:*
dtype0
Î
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
_output_shapes	
:*1

frame_name#!gradients/rnn/while/while_context*
T0
á
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
_output_shapes
	:: *
N

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
::*
T0
É
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ů
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
í
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:*
T0
§
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
Ź
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
value	B :*
_output_shapes
: *
dtype0
´
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
ˇ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"d      *
_output_shapes
:*
dtype0
š
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"d   d   *
_output_shapes
:*
dtype0
 
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
Ś
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape*
Index0*
_output_shapes
:	d*
T0
Ť
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1*
Index0*
_output_shapes

:dd*
T0
ľ
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    * 
_output_shapes
:
*
dtype0
Ń
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
is_constant( * 
_output_shapes
:
*1

frame_name#!gradients/rnn/while/while_context*
T0
ă
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*"
_output_shapes
:
: *
N

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
:
*
T0
Č
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
ü
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
*
T0
đ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
*
T0
Ĺ
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0

global_norm/L2LossL2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3

global_norm/L2Loss_1L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3

global_norm/L2Loss_2L2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3

global_norm/L2Loss_3L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3

global_norm/L2Loss_4L2Lossgradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1

global_norm/L2Loss_5L2Lossgradients/add_grad/Reshape_1*
T0*
_output_shapes
: */
_class%
#!loc:@gradients/add_grad/Reshape_1
Ő
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5*
N*

axis *
_output_shapes
:*
T0
[
global_norm/ConstConst*
valueB: *
_output_shapes
:*
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
_output_shapes
: *
dtype0
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
_output_shapes
: *
T0
Q
global_norm/global_normSqrtglobal_norm/mul*
_output_shapes
: *
T0
b
clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
ˇ
clip_by_global_norm/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
ď
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0* 
_output_shapes
:
*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
´
clip_by_global_norm/mul_2Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
ë
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
ˇ
clip_by_global_norm/mul_3Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
Č*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
ď
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0* 
_output_shapes
:
Č*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3
´
clip_by_global_norm/mul_4Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
ë
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3
ś
clip_by_global_norm/mul_5Mulgradients/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*
_output_shapes
:	d*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
Ž
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*
_output_shapes
:	d*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
Ž
clip_by_global_norm/mul_6Mulgradients/add_grad/Reshape_1clip_by_global_norm/mul*
T0*
_output_shapes	
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
¨
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*
_output_shapes	
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1
Ą
beta1_power/initial_valueConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
: *
valueB
 *fff?*
dtype0
˛
beta1_power
VariableV2*
shape: *
dtype0*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
: *
	container 
Ń
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ą
beta2_power/initial_valueConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
: *
valueB
 *wž?*
dtype0
˛
beta2_power
VariableV2*
shape: *
dtype0*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
: *
	container 
Ń
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ĺ
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
*
valueB
*    *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ň
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*
shape:
*
dtype0*
shared_name *
	container * 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ß
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
í
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ç
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
*
valueB
*    *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ô
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*
shape:
*
dtype0*
shared_name *
	container * 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ĺ
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ń
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
×
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ä
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ň
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
â
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ć
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ř
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ć
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
ĺ
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*
dtype0* 
_output_shapes
:
Č*
valueB
Č*    *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ň
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam
VariableV2*
shape:
Č*
dtype0*
shared_name *
	container * 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ß
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
í
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ç
Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*
dtype0* 
_output_shapes
:
Č*
valueB
Č*    *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ô
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1
VariableV2*
shape:
Č*
dtype0*
shared_name *
	container * 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ĺ
>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ń
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
×
Ernn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
ä
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
Ň
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
â
8rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
Ů
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
ć
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
Ř
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
ć
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
Ł
'softmax/Variable/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	d*
valueB	d*    *#
_class
loc:@softmax/Variable
°
softmax/Variable/Adam
VariableV2*
shape:	d*
dtype0*
shared_name *
	container *
_output_shapes
:	d*#
_class
loc:@softmax/Variable
Ţ
softmax/Variable/Adam/AssignAssignsoftmax/Variable/Adam'softmax/Variable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable

softmax/Variable/Adam/readIdentitysoftmax/Variable/Adam*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
Ľ
)softmax/Variable/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	d*
valueB	d*    *#
_class
loc:@softmax/Variable
˛
softmax/Variable/Adam_1
VariableV2*
shape:	d*
dtype0*
shared_name *
	container *
_output_shapes
:	d*#
_class
loc:@softmax/Variable
ä
softmax/Variable/Adam_1/AssignAssignsoftmax/Variable/Adam_1)softmax/Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable

softmax/Variable/Adam_1/readIdentitysoftmax/Variable/Adam_1*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable

)softmax/Variable_1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *%
_class
loc:@softmax/Variable_1
Ź
softmax/Variable_1/Adam
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
â
softmax/Variable_1/Adam/AssignAssignsoftmax/Variable_1/Adam)softmax/Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

softmax/Variable_1/Adam/readIdentitysoftmax/Variable_1/Adam*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
Ą
+softmax/Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *%
_class
loc:@softmax/Variable_1
Ž
softmax/Variable_1/Adam_1
VariableV2*
shape:*
dtype0*
shared_name *
	container *
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
č
 softmax/Variable_1/Adam_1/AssignAssignsoftmax/Variable_1/Adam_1+softmax/Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

softmax/Variable_1/Adam_1/readIdentitysoftmax/Variable_1/Adam_1*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
W
Adam/learning_rateConst*
valueB
 *o:*
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
 *wž?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0

FAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
use_nesterov( *
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel

DAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

FAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
use_nesterov( *
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel

DAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
ő
&Adam/update_softmax/Variable/ApplyAdam	ApplyAdamsoftmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
use_nesterov( *
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
ű
(Adam/update_softmax/Variable_1/ApplyAdam	ApplyAdamsoftmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
use_nesterov( *
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

Adam/mulMulbeta1_power/read
Adam/beta1G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
š
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias


Adam/mul_1Mulbeta2_power/read
Adam/beta2G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
˝
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

AdamNoOpG^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0

save/SaveV2/tensor_namesConst*ľ
valueŤB¨Bbeta1_powerBbeta2_powerB.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Bsoftmax/VariableBsoftmax/Variable/AdamBsoftmax/Variable/Adam_1Bsoftmax/Variable_1Bsoftmax/Variable_1/AdamBsoftmax/Variable_1/Adam_1*
_output_shapes
:*
dtype0

save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_10rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_10rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1softmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1softmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
o
save/RestoreV2/tensor_namesConst* 
valueBBbeta1_power*
_output_shapes
:*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta2_power*
_output_shapes
:*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ă
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
validate_shape(*
use_locking(*
T0*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

save/RestoreV2_2/tensor_namesConst*C
value:B8B.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ë
save/Assign_2Assign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biassave/RestoreV2_2*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

save/RestoreV2_3/tensor_namesConst*H
value?B=B3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_3Assign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adamsave/RestoreV2_3*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

save/RestoreV2_4/tensor_namesConst*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
ň
save/Assign_4Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1save/RestoreV2_4*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias

save/RestoreV2_5/tensor_namesConst*E
value<B:B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ô
save/Assign_5Assign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelsave/RestoreV2_5*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel

save/RestoreV2_6/tensor_namesConst*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_6Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adamsave/RestoreV2_6*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel

save/RestoreV2_7/tensor_namesConst*L
valueCBAB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ű
save/Assign_7Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1save/RestoreV2_7*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel

save/RestoreV2_8/tensor_namesConst*C
value:B8B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ë
save/Assign_8Assign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biassave/RestoreV2_8*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias

save/RestoreV2_9/tensor_namesConst*H
value?B=B3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_9Assign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adamsave/RestoreV2_9*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias

save/RestoreV2_10/tensor_namesConst*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
ô
save/Assign_10Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1save/RestoreV2_10*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias

save/RestoreV2_11/tensor_namesConst*E
value<B:B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
_output_shapes
:*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ö
save/Assign_11Assign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelsave/RestoreV2_11*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel

save/RestoreV2_12/tensor_namesConst*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ű
save/Assign_12Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adamsave/RestoreV2_12*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel

save/RestoreV2_13/tensor_namesConst*L
valueCBAB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ý
save/Assign_13Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1save/RestoreV2_13*
validate_shape(*
use_locking(*
T0* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
w
save/RestoreV2_14/tensor_namesConst*%
valueBBsoftmax/Variable*
_output_shapes
:*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_14Assignsoftmax/Variablesave/RestoreV2_14*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
|
save/RestoreV2_15/tensor_namesConst**
value!BBsoftmax/Variable/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_15Assignsoftmax/Variable/Adamsave/RestoreV2_15*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
~
save/RestoreV2_16/tensor_namesConst*,
value#B!Bsoftmax/Variable/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_16Assignsoftmax/Variable/Adam_1save/RestoreV2_16*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	d*#
_class
loc:@softmax/Variable
y
save/RestoreV2_17/tensor_namesConst*'
valueBBsoftmax/Variable_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_17Assignsoftmax/Variable_1save/RestoreV2_17*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
~
save/RestoreV2_18/tensor_namesConst*,
value#B!Bsoftmax/Variable_1/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_18Assignsoftmax/Variable_1/Adamsave/RestoreV2_18*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1

save/RestoreV2_19/tensor_namesConst*.
value%B#Bsoftmax/Variable_1/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_19Assignsoftmax/Variable_1/Adam_1save/RestoreV2_19*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1
ŕ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19
Ě
initNoOp8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign8^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign^softmax/Variable/Assign^softmax/Variable_1/Assign^beta1_power/Assign^beta2_power/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign^softmax/Variable/Adam/Assign^softmax/Variable/Adam_1/Assign^softmax/Variable_1/Adam/Assign!^softmax/Variable_1/Adam_1/Assign"š&őtÂ     űş	d%ÖAJç
Ę2¨2
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
ë
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

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
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
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
1
L2Loss
t"T
output"T"
Ttype:
2
7
Less
x"T
y"T
z
"
Ttype:
2		
!
LoopCond	
input


output

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
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
2
NextIteration	
data"T
output"T"	
Ttype

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
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
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
,
Sqrt
x"T
y"T"
Ttype:	
2
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
,
Tanh
x"T
y"T"
Ttype:	
2
9
TanhGrad
y"T
dy"T
z"T"
Ttype:	
2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
¸
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514Ý
W
inputsPlaceholder*
dtype0*
shape
:dd*
_output_shapes

:dd
X
targetsPlaceholder*
dtype0*
shape
:dd*
_output_shapes

:dd
N
	keep_probPlaceholder*
dtype0*
shape:*
_output_shapes
:
]
DropoutWrapperInit/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
_
DropoutWrapperInit/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
_
DropoutWrapperInit_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
a
DropoutWrapperInit_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?

JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
dtype0*
valueB:d

PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
ý
KMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N

PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    

JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillKMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatPMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:dd

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
dtype0*
valueB:d

RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

MMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
 
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillMMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:dd

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
dtype0*
valueB:d

LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
dtype0*
valueB:d

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
dtype0*
valueB:d

RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

MMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N

RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
 
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFillMMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatRMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes

:dd

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
dtype0*
valueB:d

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
dtype0*
valueB:d

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
dtype0*
valueB:d

TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

OMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV2NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ś
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1FillOMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes

:dd

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
dtype0*
valueB:d

NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
dtype0*
valueB:d
U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
P
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :

one_hotOneHotinputsone_hot/depthone_hot/on_valueone_hot/off_value*
TI0*
T0*
axis˙˙˙˙˙˙˙˙˙*#
_output_shapes
:dd
F
RankConst*
_output_shapes
: *
dtype0*
value	B :
M
range/startConst*
_output_shapes
: *
dtype0*
value	B :
M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
V
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
`
concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
q
concatConcatV2concat/values_0rangeconcat/axis*
T0*

Tidx0*
_output_shapes
:*
N
b
	transpose	Transposeone_hotconcat*
T0*
Tperm0*#
_output_shapes
:dd
^
	rnn/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d      
a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*

begin_mask *
shrink_axis_mask*
ellipsis_mask *
T0*
end_mask *
new_axis_mask *
_output_shapes
: 
S
	rnn/ConstConst*
_output_shapes
:*
dtype0*
valueB:d
U
rnn/Const_1Const*
_output_shapes
:*
dtype0*
valueB:d
Q
rnn/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y

rnn/concatConcatV2	rnn/Constrnn/Const_1rnn/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
W
	rnn/zerosFill
rnn/concatrnn/zeros/Const*
T0*
_output_shapes

:dd
J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Ň
rnn/TensorArrayTensorArrayV3rnn/strided_slice*/
tensor_array_namernn/dynamic_rnn/output_0*
element_shape:*
clear_after_read(*
dtype0*
dynamic_size( *
_output_shapes

:: 
Ó
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*.
tensor_array_namernn/dynamic_rnn/input_0*
element_shape:*
clear_after_read(*
dtype0*
dynamic_size( *
_output_shapes

:: 
q
rnn/TensorArrayUnstack/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d      
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ě
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*

begin_mask *
shrink_axis_mask*
ellipsis_mask *
T0*
end_mask *
new_axis_mask *
_output_shapes
: 
d
"rnn/TensorArrayUnstack/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
d
"rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Ä
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
_output_shapes
: *
_class
loc:@transpose*
T0

rnn/while/EnterEnterrnn/time*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes
: 
Ľ
rnn/while/Enter_1Enterrnn/TensorArray:1*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes
: 
ć
rnn/while/Enter_2EnterJMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes

:dd
č
rnn/while/Enter_3EnterLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes

:dd
č
rnn/while/Enter_4EnterLMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes

:dd
ę
rnn/while/Enter_5EnterNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes

:dd
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
N*
_output_shapes
: : *
T0
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0
|
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
N* 
_output_shapes
:dd: *
T0
|
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
N* 
_output_shapes
:dd: *
T0
|
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
N* 
_output_shapes
:dd: *
T0
|
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
N* 
_output_shapes
:dd: *
T0
¨
rnn/while/Less/EnterEnterrnn/strided_slice*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
: 
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0*
_output_shapes
: 
F
rnn/while/LoopCondLoopCondrnn/while/Less*
_output_shapes
: 

rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge*
T0

rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1*
T0

rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_2*
T0

rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_3*
T0

rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_4*
T0

rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_5*
T0
S
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0*
_output_shapes
: 
_
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0*
_output_shapes

:dd
_
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0*
_output_shapes

:dd
_
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0*
_output_shapes

:dd
_
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
T0*
_output_shapes

:dd
š
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ä
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
: 
ş
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	d
ç
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
:*
valueB"     *
dtype0
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *ÝĂ˝*
dtype0
Ů
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *ÝĂ=*
dtype0
Ó
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0*

seed *
T0*
seed2 * 
_output_shapes
:

Ţ
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ň
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
ä
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
í
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
*
dtype0*
shared_name * 
_output_shapes
:
*
	container 
Ů
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0

5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

Ň
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ß
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ă
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
_output_shapes	
:

Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :

Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
T0*

Tidx0*
_output_shapes
:	d*
N

Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context* 
_output_shapes
:

§
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	d

Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes	
:

Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0*
_output_shapes
:	d

@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Ą
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
ą
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
	num_split*
T0*<
_output_shapes*
(:dd:dd:dd:dd

@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
ô
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
T0*
_output_shapes

:dd
ś
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0*
_output_shapes

:dd
Č
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMulrnn/while/Identity_2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:dd
ş
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0*
_output_shapes

:dd
´
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0*
_output_shapes

:dd
÷
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
T0*
_output_shapes

:dd
ň
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
T0*
_output_shapes

:dd
´
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:dd
ź
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
T0*
_output_shapes

:dd
ů
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:dd

1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
dtype0*
valueB"d   d   

>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *    

>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ű
Hrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
T0*

seed *
_output_shapes

:dd*
seed2 *
dtype0
ć
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes
: 
ř
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
T0*
_output_shapes

:dd
ę
:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
T0*
_output_shapes

:dd
Ă
5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/EnterEnter	keep_prob*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ě
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
T0*
_output_shapes
:

1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
T0*
_output_shapes
:
Ö
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
T0*
_output_shapes
:
Ă
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
_output_shapes

:dd
ç
Qrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
_output_shapes
:*
valueB"Č     *
dtype0
Ů
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *ÍĚĚ˝*
dtype0
Ů
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0
Ó
Yrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shape*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0*

seed *
T0*
seed2 * 
_output_shapes
:
Č
Ţ
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ň
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
Č*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
ä
Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
Č*
T0*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
í
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
Č*
dtype0*
shared_name * 
_output_shapes
:
Č*
	container 
Ů
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0

5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
Č
Ň
@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/ConstConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ß
.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ă
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0

3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
_output_shapes	
:

Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :

Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatConcatV2/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulrnn/while/Identity_5Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axis*
T0*

Tidx0*
_output_shapes
:	dČ*
N

Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context* 
_output_shapes
:
Č
§
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	d

Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes	
:

Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0*
_output_shapes
:	d

@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Ą
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
ą
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd*
	num_split*
T0*<
_output_shapes*
(:dd:dd:dd:dd

@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
ô
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/y*
T0*
_output_shapes

:dd
ś
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add*
T0*
_output_shapes

:dd
Č
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mulMulrnn/while/Identity_4Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
T0*
_output_shapes

:dd
ş
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split*
T0*
_output_shapes

:dd
´
?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:1*
T0*
_output_shapes

:dd
÷
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
T0*
_output_shapes

:dd
ň
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1*
T0*
_output_shapes

:dd
´
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
T0*
_output_shapes

:dd
ź
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:3*
T0*
_output_shapes

:dd
ů
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes

:dd

1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
dtype0*
valueB"d   d   

>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *    

>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ű
Hrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Shape*
T0*

seed *
_output_shapes

:dd*
seed2 *
dtype0
ć
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
ř
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/sub*
T0*
_output_shapes

:dd
ę
:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
T0*
_output_shapes

:dd
Ě
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform*
T0*
_output_shapes
:

1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/add*
T0*
_output_shapes
:
Ö
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
T0*
_output_shapes
:
Ă
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
T0*
_output_shapes

:dd

3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
parallel_iterations *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¸
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulrnn/while/Identity_1*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
f
rnn/while/add/yConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0*
_output_shapes
: 
X
rnn/while/NextIterationNextIterationrnn/while/add*
T0*
_output_shapes
: 
z
rnn/while/NextIteration_1NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

rnn/while/NextIteration_2NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0*
_output_shapes

:dd

rnn/while/NextIteration_3NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
T0*
_output_shapes

:dd

rnn/while/NextIteration_4NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
T0*
_output_shapes

:dd

rnn/while/NextIteration_5NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
T0*
_output_shapes

:dd
I
rnn/while/ExitExitrnn/while/Switch*
T0*
_output_shapes
: 
M
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0*
_output_shapes
: 
U
rnn/while/Exit_2Exitrnn/while/Switch_2*
T0*
_output_shapes

:dd
U
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0*
_output_shapes

:dd
U
rnn/while/Exit_4Exitrnn/while/Switch_4*
T0*
_output_shapes

:dd
U
rnn/while/Exit_5Exitrnn/while/Switch_5*
T0*
_output_shapes

:dd

&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*"
_class
loc:@rnn/TensorArray*
_output_shapes
: 

 rnn/TensorArrayStack/range/startConst*
_output_shapes
: *
dtype0*
value	B : *"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*"
_class
loc:@rnn/TensorArray
ä
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@rnn/TensorArray*

Tidx0
đ
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*"
_output_shapes
:ddd*
element_shape
:dd*"
_class
loc:@rnn/TensorArray*
dtype0
\
rnn/Const_2Const*
_output_shapes
:*
dtype0*
valueB"d   d   
U
rnn/Const_3Const*
_output_shapes
:*
dtype0*
valueB:d
J
rnn/RankConst*
_output_shapes
: *
dtype0*
value	B :
Q
rnn/range/startConst*
_output_shapes
: *
dtype0*
value	B :
Q
rnn/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0*
_output_shapes
:
f
rnn/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
S
rnn/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N

rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
T0*
Tperm0*"
_output_shapes
:ddd
U
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :
P
concat_1Identityrnn/transpose*
T0*"
_output_shapes
:ddd
^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d   
c
ReshapeReshapeconcat_1Reshape/shape*
Tshape0*
T0*
_output_shapes
:	Nd
o
softmax/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"d      
b
softmax/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
softmax/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ=
Ť
(softmax/truncated_normal/TruncatedNormalTruncatedNormalsoftmax/truncated_normal/shape*
T0*

seed *
_output_shapes
:	d*
seed2 *
dtype0

softmax/truncated_normal/mulMul(softmax/truncated_normal/TruncatedNormalsoftmax/truncated_normal/stddev*
T0*
_output_shapes
:	d

softmax/truncated_normalAddsoftmax/truncated_normal/mulsoftmax/truncated_normal/mean*
T0*
_output_shapes
:	d

softmax/Variable
VariableV2*
shared_name *
_output_shapes
:	d*
	container *
shape:	d*
dtype0
Ĺ
softmax/Variable/AssignAssignsoftmax/Variablesoftmax/truncated_normal*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0

softmax/Variable/readIdentitysoftmax/Variable*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
\
softmax/zerosConst*
_output_shapes	
:*
dtype0*
valueB*    

softmax/Variable_1
VariableV2*
shared_name *
_output_shapes	
:*
	container *
shape:*
dtype0
ź
softmax/Variable_1/AssignAssignsoftmax/Variable_1softmax/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

softmax/Variable_1/readIdentitysoftmax/Variable_1*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

MatMulMatMulReshapesoftmax/Variable/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
N
V
addAddMatMulsoftmax/Variable_1/read*
T0* 
_output_shapes
:
N
F
predictionsSoftmaxadd*
T0* 
_output_shapes
:
N
W
one_hot_1/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
X
one_hot_1/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
R
one_hot_1/depthConst*
_output_shapes
: *
dtype0*
value
B :
Ł
	one_hot_1OneHottargetsone_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
TI0*
T0*
axis˙˙˙˙˙˙˙˙˙*#
_output_shapes
:dd
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"'     
i
	Reshape_1Reshape	one_hot_1Reshape_1/shape*
Tshape0*
T0* 
_output_shapes
:
N
H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"'     
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"'     
G
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
:
SubSubRank_2Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
_output_shapes
:*

axis *
T0*
N
T

Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
T0*
_output_shapes
:
d
concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
w
concat_2ConcatV2concat_2/values_0Sliceconcat_2/axis*
T0*

Tidx0*
_output_shapes
:*
N
\
	Reshape_2Reshapeaddconcat_2*
Tshape0*
T0* 
_output_shapes
:
N
H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :
X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"'     
I
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_1SubRank_3Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
_output_shapes
:*

axis *
T0*
N
V
Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
Index0*
T0*
_output_shapes
:
d
concat_3/values_0Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
y
concat_3ConcatV2concat_3/values_0Slice_1concat_3/axis*
T0*

Tidx0*
_output_shapes
:*
N
b
	Reshape_3Reshape	Reshape_1concat_3*
Tshape0*
T0* 
_output_shapes
:
N

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*
T0*'
_output_shapes
:N:
N
I
Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
>
Sub_2SubRank_1Sub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: 
U
Slice_2/sizePackSub_2*
_output_shapes
:*

axis *
T0*
N
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
Tshape0*
T0*
_output_shapes	
:N
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
\
MeanMean	Reshape_4Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Z
batch_loss/tagsConst*
_output_shapes
: *
dtype0*
valueB B
batch_loss
S

batch_lossScalarSummarybatch_loss/tagsMean*
T0*
_output_shapes
: 
O
Merge/MergeSummaryMergeSummary
batch_loss*
N*
_output_shapes
: 
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
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
S
gradients/f_countConst*
_output_shapes
: *
dtype0*
value	B : 
§
gradients/f_count_1Entergradients/f_count*
parallel_iterations *
is_constant( *
T0*'

frame_namernn/while/while_context*
_output_shapes
: 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*
_output_shapes
: : *
T0
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
T0*
_output_shapes
: : 
f
gradients/Add/yConst^rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
á
gradients/NextIterationNextIterationgradients/Add[^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2^^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2f^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2O^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2^^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2f^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
_output_shapes
: *
dtype0*
value	B :
ł
gradients/b_count_1Entergradients/f_count_2*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*
_output_shapes
: : *
T0
ş
gradients/GreaterEqual/EnterEntergradients/b_count*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
˛
gradients/NextIteration_1NextIterationgradients/SubV^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
m
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:N

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes	
:N
d
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 

gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
Â
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0

gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: *,
_class"
 loc:@gradients/Mean_grad/Shape
Č
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*,
_class"
 loc:@gradients/Mean_grad/Shape*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0

gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape
°
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
Ž
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes	
:N
i
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N

 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
Tshape0*
T0*
_output_shapes	
:N
m
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0* 
_output_shapes
:
N

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ú
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	N
ź
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0* 
_output_shapes
:
N
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"'     
ś
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape*
Tshape0*
T0* 
_output_shapes
:
N
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"'     
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0* 
_output_shapes
:
N
­
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:
Š
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/Reshapesoftmax/Variable/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	Nd

gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	d
q
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d   d   d   
 
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
Tshape0*
T0*"
_output_shapes
:ddd
v
.gradients/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/concat_1*
T0*
_output_shapes
:
˝
&gradients/rnn/transpose_grad/transpose	Transposegradients/Reshape_grad/Reshape.gradients/rnn/transpose_grad/InvertPermutation*
T0*
Tperm0*"
_output_shapes
:ddd
ę
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_1*
source	gradients*"
_class
loc:@rnn/TensorArray*
_output_shapes

:: 

Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_1Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
T0

_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range&gradients/rnn/transpose_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
d
gradients/zerosConst*
_output_shapes

:dd*
dtype0*
valueBdd*    
f
gradients/zeros_1Const*
_output_shapes

:dd*
dtype0*
valueBdd*    
f
gradients/zeros_2Const*
_output_shapes

:dd*
dtype0*
valueBdd*    
f
gradients/zeros_3Const*
_output_shapes

:dd*
dtype0*
valueBdd*    

&gradients/rnn/while/Exit_1_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
Ę
&gradients/rnn/while/Exit_2_grad/b_exitEntergradients/zeros*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes

:dd
Ě
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_1*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes

:dd
Ě
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_2*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes

:dd
Ě
&gradients/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_3*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes

:dd
ş
*gradients/rnn/while/Switch_1_grad/b_switchMerge&gradients/rnn/while/Exit_1_grad/b_exit1gradients/rnn/while/Switch_1_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
Â
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration*
N* 
_output_shapes
:dd: *
T0
Â
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration*
N* 
_output_shapes
:dd: *
T0
Â
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration*
N* 
_output_shapes
:dd: *
T0
Â
*gradients/rnn/while/Switch_5_grad/b_switchMerge&gradients/rnn/while/Exit_5_grad/b_exit1gradients/rnn/while/Switch_5_grad_1/NextIteration*
N* 
_output_shapes
:dd: *
T0
Ô
'gradients/rnn/while/Merge_1_grad/SwitchSwitch*gradients/rnn/while/Switch_1_grad/b_switchgradients/b_count_2*
_output_shapes
: : *=
_class3
1/loc:@gradients/rnn/while/Switch_1_grad/b_switch*
T0
ä
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
T0
ä
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
T0
ä
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
T0
ä
'gradients/rnn/while/Merge_5_grad/SwitchSwitch*gradients/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
T0
w
%gradients/rnn/while/Enter_1_grad/ExitExit'gradients/rnn/while/Merge_1_grad/Switch*
T0*
_output_shapes
: 

%gradients/rnn/while/Enter_2_grad/ExitExit'gradients/rnn/while/Merge_2_grad/Switch*
T0*
_output_shapes

:dd

%gradients/rnn/while/Enter_3_grad/ExitExit'gradients/rnn/while/Merge_3_grad/Switch*
T0*
_output_shapes

:dd

%gradients/rnn/while/Enter_4_grad/ExitExit'gradients/rnn/while/Merge_4_grad/Switch*
T0*
_output_shapes

:dd

%gradients/rnn/while/Enter_5_grad/ExitExit'gradients/rnn/while/Merge_5_grad/Switch*
T0*
_output_shapes

:dd
Č
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
parallel_iterations *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ý
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter)gradients/rnn/while/Merge_1_grad/Switch:1*
source	gradients*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
_output_shapes

:: 
×
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity)gradients/rnn/while/Merge_1_grad/Switch:1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
Ď
]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*%
_class
loc:@rnn/while/Identity
¤
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*%
_class
loc:@rnn/while/Identity*
	elem_type0*

stack_name *
_output_shapes
:
Ż
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:

Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity^gradients/Add*
swap_memory( *
T0*
_output_shapes
: 
Ä
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
Ő
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
§
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes

:dd
ź
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape
â
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*
	elem_type0*

stack_name *
_output_shapes
:
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
â
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1
č
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*
	elem_type0*

stack_name *
_output_shapes
:
ż
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
č
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
â
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_size*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
	elem_type0*

stack_name *
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor^gradients/Add*
swap_memory( *
T0*
_output_shapes
:
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ë
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulMulNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2*
T0*
_output_shapes
:
Ł
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
T0*
_output_shapes
:
â
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div
­
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_size*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
	elem_type0*

stack_name *
_output_shapes
:

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
˘
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div^gradients/Add*
swap_memory( *
T0*
_output_shapes
:
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ď
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*
_output_shapes
:
Š
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
Tshape0*
T0*
_output_shapes
:
Ľ
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
Ů
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1
ä
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*
	elem_type0*

stack_name *
_output_shapes
:
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ä
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ä
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/EnterEnter	keep_prob*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
§
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
ń
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2
ş
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_size*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
	elem_type0*

stack_name *
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ľ
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ń
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Á
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2*
T0*
_output_shapes

:dd

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
ţ
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
T0*
_output_shapes
:

1gradients/rnn/while/Switch_1_grad_1/NextIterationNextIteration)gradients/rnn/while/Merge_1_grad/Switch:1*
T0*
_output_shapes
: 
ú
gradients/AddNAddN)gradients/rnn/while/Merge_5_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
_output_shapes

:dd*
N
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2
ŕ
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ű
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ó
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1
á
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*
	elem_type0*

stack_name *
_output_shapes
:
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ü
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
÷
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN*
T0*
_output_shapes

:dd
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
É
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape*
T0*
_output_shapes

:dd
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1*
T0*
_output_shapes

:dd

gradients/AddN_1AddN)gradients/rnn/while/Merge_4_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
_output_shapes

:dd*
N
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid
Ú
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_size*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
	elem_type0*

stack_name *
_output_shapes
:
ľ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ő
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ę
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
¸
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Đ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
Ö
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*'
_class
loc:@rnn/while/Identity_4
°
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*'
_class
loc:@rnn/while/Identity_4*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ť
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_4^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh
Ű
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ö
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ž
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1
ä
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*
	elem_type0*

stack_name *
_output_shapes
:
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ß
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Â
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1*
T0*
_output_shapes

:dd
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
Ě
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1*
T0*
_output_shapes

:dd
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape*
T0*
_output_shapes

:dd
Ç
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1*
T0*
_output_shapes

:dd
Â
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape*
T0*
_output_shapes

:dd
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¨
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ü
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
ŕ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
˝
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ž
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
ń
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/Const*
T0*

Tidx0*
_output_shapes
:	d*
N

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
Š
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:
Č
č
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	dČ

hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat
é
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*
	elem_type0*

stack_name *
_output_shapes
:
Ĺ
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ĺ
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	dČ
Ú
jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
 
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	dČ
ň
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
Č
Ź
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:*
dtype0*
valueB*    
Î
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes	
:
á
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
É
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ů
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
í
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
§
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
Ź
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
´
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ˇ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
š
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
 
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
Ľ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes

:dd
Ť
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes

:dd
ľ
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
Č*
dtype0*
valueB
Č*    
Ń
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:
Č
ă
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
Č: *
T0

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
Č:
Č
Č
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Č
ü
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
Č
đ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
Č
ź
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape
â
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
	elem_type0*

stack_name *
_output_shapes
:
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
â
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1
č
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
	elem_type0*

stack_name *
_output_shapes
:
ż
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
č
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
â
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
	elem_type0*

stack_name *
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^gradients/Add*
swap_memory( *
T0*
_output_shapes
:
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ë
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
T0*
_output_shapes
:
Ł
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
T0*
_output_shapes
:
â
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div
­
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
	elem_type0*

stack_name *
_output_shapes
:

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
˘
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^gradients/Add*
swap_memory( *
T0*
_output_shapes
:
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ď
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:

Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice*
T0*
_output_shapes
:
Š
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
Tshape0*
T0*
_output_shapes
:
Ľ
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
Ů
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1
ä
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
	elem_type0*

stack_name *
_output_shapes
:
ť
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ä
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1^gradients/Add*
swap_memory( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
§
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
ń
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2
ş
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
	elem_type0*

stack_name *
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ľ
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ń
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Á
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
T0*
_output_shapes

:dd

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
ţ
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ť
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*
T0*
_output_shapes
:
Ĺ
1gradients/rnn/while/Switch_5_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes

:dd
ü
gradients/AddN_2AddN)gradients/rnn/while/Merge_3_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
_output_shapes

:dd*
N
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2
ŕ
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ű
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ő
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN_2^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1
á
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
	elem_type0*

stack_name *
_output_shapes
:
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ü
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ů
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN_2*
T0*
_output_shapes

:dd
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
É
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
T0*
_output_shapes

:dd
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1*
T0*
_output_shapes

:dd

gradients/AddN_3AddN)gradients/rnn/while/Merge_2_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
_output_shapes

:dd*
N
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_3egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_3ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid
Ú
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
	elem_type0*

stack_name *
_output_shapes
:
ľ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ő
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ę
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
¸
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Đ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
Ö
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*'
_class
loc:@rnn/while/Identity_2
°
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*'
_class
loc:@rnn/while/Identity_2*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ť
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_2^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ź
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
ś
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¸
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
ë
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh
Ű
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
	elem_type0*

stack_name *
_output_shapes
:
š
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
Ö
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Î
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
ž
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
T0*
_output_shapes

:dd
Ö
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ĺ
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd

dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1
ä
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
	elem_type0*

stack_name *
_output_shapes
:
˝
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ß
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^gradients/Add*
swap_memory( *
T0*
_output_shapes

:dd
Ň
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes

:dd
Â
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1*
T0*
_output_shapes

:dd
Ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ë
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes

:dd
Ě
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1*
T0*
_output_shapes

:dd
Ň
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
T0*
_output_shapes

:dd
Ç
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1*
T0*
_output_shapes

:dd
Â
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
T0*
_output_shapes

:dd
´
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
¨
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
dtype0*
valueB 
ĺ
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ü
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ż
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
Tshape0*
T0*
_output_shapes

:dd
ŕ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
˝
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ž
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
ń
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*
T0*

Tidx0*
_output_shapes
:	d*
N

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes	
:
Š
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

č
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	d

hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat
é
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
	elem_type0*

stack_name *
_output_shapes
:
Ĺ
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*'

frame_namernn/while/while_context*
_output_shapes
:
ĺ
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^gradients/Add*
swap_memory( *
T0*
_output_shapes
:	d
Ú
jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
is_constant(*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
 
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	d
ň
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:

Ź
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:*
dtype0*
valueB*    
Î
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes	
:
á
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:: *
T0

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
É
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ů
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
í
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:
§
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
Ź
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
dtype0*
value	B :
´
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ˇ
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d      
š
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
dtype0*
valueB"d   d   
 
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::
Ś
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes
:	d
Ť
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes

:dd
ľ
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
*
dtype0*
valueB
*    
Ń
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations *
is_constant( *
T0*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

ă
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
: *
T0

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

Č
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

ü
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

đ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

Ĺ
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
T0*
_output_shapes

:dd

global_norm/L2LossL2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_1L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_2L2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_3L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0

global_norm/L2Loss_4L2Lossgradients/MatMul_grad/MatMul_1*
_output_shapes
: *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0

global_norm/L2Loss_5L2Lossgradients/add_grad/Reshape_1*
_output_shapes
: */
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
Ő
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5*
_output_shapes
:*

axis *
T0*
N
[
global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
X
global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
ˇ
clip_by_global_norm/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
ď
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1* 
_output_shapes
:
*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
´
clip_by_global_norm/mul_2Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ë
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ˇ
clip_by_global_norm/mul_3Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
Č*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
ď
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3* 
_output_shapes
:
Č*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
´
clip_by_global_norm/mul_4Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ë
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
_output_shapes	
:*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
ś
clip_by_global_norm/mul_5Mulgradients/MatMul_grad/MatMul_1clip_by_global_norm/mul*
_output_shapes
:	d*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
Ž
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	d*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
Ž
clip_by_global_norm/mul_6Mulgradients/add_grad/Reshape_1clip_by_global_norm/mul*
_output_shapes	
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
¨
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
_output_shapes	
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
Ą
beta1_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
˛
beta1_power
VariableV2*
	container *
shape: *
dtype0*
shared_name *
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ń
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

beta1_power/readIdentitybeta1_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
Ą
beta2_power/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
˛
beta2_power
VariableV2*
	container *
shape: *
dtype0*
shared_name *
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
Ń
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

beta2_power/readIdentitybeta2_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
ĺ
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
valueB
*    *
dtype0
ň
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
*
dtype0*
shared_name * 
_output_shapes
:
*
	container 
ß
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
í
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
ç
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
*
valueB
*    *
dtype0
ô
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
*
dtype0*
shared_name * 
_output_shapes
:
*
	container 
ĺ
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
ń
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
×
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ä
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ň
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
â
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
Ů
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ć
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ř
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
ć
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
ĺ
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zerosConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
Č*
valueB
Č*    *
dtype0
ň
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
Č*
dtype0*
shared_name * 
_output_shapes
:
Č*
	container 
ß
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
í
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
ç
Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
Č*
valueB
Č*    *
dtype0
ô
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1
VariableV2*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
Č*
dtype0*
shared_name * 
_output_shapes
:
Č*
	container 
ĺ
>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
ń
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
×
Ernn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ä
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ň
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
â
8rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
Ů
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:*
valueB*    *
dtype0
ć
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1
VariableV2*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
Ř
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
ć
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
Ł
'softmax/Variable/Adam/Initializer/zerosConst*#
_class
loc:@softmax/Variable*
_output_shapes
:	d*
valueB	d*    *
dtype0
°
softmax/Variable/Adam
VariableV2*#
_class
loc:@softmax/Variable*
shape:	d*
dtype0*
shared_name *
_output_shapes
:	d*
	container 
Ţ
softmax/Variable/Adam/AssignAssignsoftmax/Variable/Adam'softmax/Variable/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0

softmax/Variable/Adam/readIdentitysoftmax/Variable/Adam*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
Ľ
)softmax/Variable/Adam_1/Initializer/zerosConst*#
_class
loc:@softmax/Variable*
_output_shapes
:	d*
valueB	d*    *
dtype0
˛
softmax/Variable/Adam_1
VariableV2*#
_class
loc:@softmax/Variable*
shape:	d*
dtype0*
shared_name *
_output_shapes
:	d*
	container 
ä
softmax/Variable/Adam_1/AssignAssignsoftmax/Variable/Adam_1)softmax/Variable/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0

softmax/Variable/Adam_1/readIdentitysoftmax/Variable/Adam_1*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0

)softmax/Variable_1/Adam/Initializer/zerosConst*%
_class
loc:@softmax/Variable_1*
_output_shapes	
:*
valueB*    *
dtype0
Ź
softmax/Variable_1/Adam
VariableV2*%
_class
loc:@softmax/Variable_1*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
â
softmax/Variable_1/Adam/AssignAssignsoftmax/Variable_1/Adam)softmax/Variable_1/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

softmax/Variable_1/Adam/readIdentitysoftmax/Variable_1/Adam*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0
Ą
+softmax/Variable_1/Adam_1/Initializer/zerosConst*%
_class
loc:@softmax/Variable_1*
_output_shapes	
:*
valueB*    *
dtype0
Ž
softmax/Variable_1/Adam_1
VariableV2*%
_class
loc:@softmax/Variable_1*
shape:*
dtype0*
shared_name *
_output_shapes	
:*
	container 
č
 softmax/Variable_1/Adam_1/AssignAssignsoftmax/Variable_1/Adam_1+softmax/Variable_1/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

softmax/Variable_1/Adam_1/readIdentitysoftmax/Variable_1/Adam_1*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o:
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
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2

FAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
use_nesterov( * 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0

DAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
use_nesterov( *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

FAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_locking( *
use_nesterov( * 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0

DAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_locking( *
use_nesterov( *
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
ő
&Adam/update_softmax/Variable/ApplyAdam	ApplyAdamsoftmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_locking( *
use_nesterov( *
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
ű
(Adam/update_softmax/Variable_1/ApplyAdam	ApplyAdamsoftmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
use_nesterov( *
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

Adam/mulMulbeta1_power/read
Adam/beta1G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
š
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
use_locking( *
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0


Adam/mul_1Mulbeta2_power/read
Adam/beta2G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
˝
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
use_locking( *
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

AdamNoOpG^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
_output_shapes
: *
dtype0*
valueB Bmodel

save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*ľ
valueŤB¨Bbeta1_powerBbeta2_powerB.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Bsoftmax/VariableBsoftmax/Variable/AdamBsoftmax/Variable/Adam_1Bsoftmax/Variable_1Bsoftmax/Variable_1/AdamBsoftmax/Variable_1/Adam_1

save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_10rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_10rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1softmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1softmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
_class
loc:@save/Const*
T0
o
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta1_power
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
ż
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
q
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0* 
valueBBbeta2_power
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ă
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
validate_shape(*
use_locking(*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*
dtype0*C
value:B8B.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ë
save/Assign_2Assign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biassave/RestoreV2_2*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*
dtype0*H
value?B=B3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_3Assign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adamsave/RestoreV2_3*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*
dtype0*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
ň
save/Assign_4Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1save/RestoreV2_4*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0

save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*
dtype0*E
value<B:B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ô
save/Assign_5Assign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelsave/RestoreV2_5*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0

save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*
dtype0*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_6Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adamsave/RestoreV2_6*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0

save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ű
save/Assign_7Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1save/RestoreV2_7*
validate_shape(*
use_locking(* 
_output_shapes
:
*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0

save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*
dtype0*C
value:B8B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ë
save/Assign_8Assign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biassave/RestoreV2_8*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0

save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*
dtype0*H
value?B=B3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_9Assign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adamsave/RestoreV2_9*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0

save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*
dtype0*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
ô
save/Assign_10Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1save/RestoreV2_10*
validate_shape(*
use_locking(*
_output_shapes	
:*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0

save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*
dtype0*E
value<B:B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ö
save/Assign_11Assign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelsave/RestoreV2_11*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0

save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*
dtype0*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ű
save/Assign_12Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adamsave/RestoreV2_12*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0

save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*
dtype0*L
valueCBAB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ý
save/Assign_13Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1save/RestoreV2_13*
validate_shape(*
use_locking(* 
_output_shapes
:
Č*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
w
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*
dtype0*%
valueBBsoftmax/Variable
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_14Assignsoftmax/Variablesave/RestoreV2_14*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
|
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*
dtype0**
value!BBsoftmax/Variable/Adam
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_15Assignsoftmax/Variable/Adamsave/RestoreV2_15*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
~
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bsoftmax/Variable/Adam_1
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_16Assignsoftmax/Variable/Adam_1save/RestoreV2_16*
validate_shape(*
use_locking(*
_output_shapes
:	d*#
_class
loc:@softmax/Variable*
T0
y
save/RestoreV2_17/tensor_namesConst*
_output_shapes
:*
dtype0*'
valueBBsoftmax/Variable_1
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save/Assign_17Assignsoftmax/Variable_1save/RestoreV2_17*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0
~
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*
dtype0*,
value#B!Bsoftmax/Variable_1/Adam
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_18Assignsoftmax/Variable_1/Adamsave/RestoreV2_18*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0

save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*
dtype0*.
value%B#Bsoftmax/Variable_1/Adam_1
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save/Assign_19Assignsoftmax/Variable_1/Adam_1save/RestoreV2_19*
validate_shape(*
use_locking(*
_output_shapes	
:*%
_class
loc:@softmax/Variable_1*
T0
ŕ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19
Ě
initNoOp8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign8^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign^softmax/Variable/Assign^softmax/Variable_1/Assign^beta1_power/Assign^beta2_power/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign^softmax/Variable/Adam/Assign^softmax/Variable/Adam_1/Assign^softmax/Variable_1/Adam/Assign!^softmax/Variable_1/Adam_1/Assign""
	summaries

batch_loss:0"
train_op

Adam"ÜĄ
while_contextÉĄĹĄ
ÁĄ
rnn/while/while_context *rnn/while/LoopCond:02rnn/while/Merge:0:rnn/while/Identity:0Brnn/while/Exit:0Brnn/while/Exit_1:0Brnn/while/Exit_2:0Brnn/while/Exit_3:0Brnn/while/Exit_4:0Brnn/while/Exit_5:0Bgradients/f_count_2:0JÁ
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
\gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/Enter:0
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/Enter:0
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/Enter:0
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2:0
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/Enter:0
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc:0
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter:0
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape:0
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter:0
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/Enter:0
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/Enter:0
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2:0
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/Enter:0
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2:0
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enter:0
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2:0
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/Enter:0
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc:0
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter:0
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape:0
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter:0
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc:0
keep_prob:0
rnn/TensorArray:0
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn/TensorArray_1:0
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:0
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:0
rnn/strided_slice:0
rnn/while/Enter:0
rnn/while/Enter_1:0
rnn/while/Enter_2:0
rnn/while/Enter_3:0
rnn/while/Enter_4:0
rnn/while/Enter_5:0
rnn/while/Exit:0
rnn/while/Exit_1:0
rnn/while/Exit_2:0
rnn/while/Exit_3:0
rnn/while/Exit_4:0
rnn/while/Exit_5:0
rnn/while/Identity:0
rnn/while/Identity_1:0
rnn/while/Identity_2:0
rnn/while/Identity_3:0
rnn/while/Identity_4:0
rnn/while/Identity_5:0
rnn/while/Less/Enter:0
rnn/while/Less:0
rnn/while/LoopCond:0
rnn/while/Merge:0
rnn/while/Merge:1
rnn/while/Merge_1:0
rnn/while/Merge_1:1
rnn/while/Merge_2:0
rnn/while/Merge_2:1
rnn/while/Merge_3:0
rnn/while/Merge_3:1
rnn/while/Merge_4:0
rnn/while/Merge_4:1
rnn/while/Merge_5:0
rnn/while/Merge_5:1
rnn/while/NextIteration:0
rnn/while/NextIteration_1:0
rnn/while/NextIteration_2:0
rnn/while/NextIteration_3:0
rnn/while/NextIteration_4:0
rnn/while/NextIteration_5:0
rnn/while/Switch:0
rnn/while/Switch:1
rnn/while/Switch_1:0
rnn/while/Switch_1:1
rnn/while/Switch_2:0
rnn/while/Switch_2:1
rnn/while/Switch_3:0
rnn/while/Switch_3:1
rnn/while/Switch_4:0
rnn/while/Switch_4:1
rnn/while/Switch_5:0
rnn/while/Switch_5:1
#rnn/while/TensorArrayReadV3/Enter:0
%rnn/while/TensorArrayReadV3/Enter_1:0
rnn/while/TensorArrayReadV3:0
5rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn/while/add/y:0
rnn/while/add:0
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Const:0
Irnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul:0
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1:0
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2:0
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh:0
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y:0
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1:0
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis:0
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat:0
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2:0
Lrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dim:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:0
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3
3rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor:0
3rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape:0
7rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:0
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add:0
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div:0
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul:0
Jrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform:0
@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max:0
@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min:0
@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul:0
@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub:0
<rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform:0
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter:0
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Const:0
Irnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter:0
Crnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul:0
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1:0
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2:0
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh:0
Crnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/y:0
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1:0
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axis:0
Crnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat:0
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2:0
Lrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dim:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:0
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:1
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:2
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:3
3rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor:0
3rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Shape:0
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/add:0
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div:0
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul:0
Jrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniform:0
@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/max:0
@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min:0
@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mul:0
@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/sub:0
<rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform:0ž
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/Enter:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter:0ź
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter:0Ŕ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0ž
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/Enter:0-
rnn/strided_slice:0rnn/while/Less/Enter:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter:0:
rnn/TensorArray_1:0#rnn/while/TensorArrayReadV3/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter:0ž
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/Enter:0i
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%rnn/while/TensorArrayReadV3/Enter_1:0ś
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc:0Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/Enter:0Ć
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0ź
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/Enter:0ś
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc:0Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/Enter:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enter:0Ŕ
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0ž
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/Enter:0F
keep_prob:07rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:0°
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter:0
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:0Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/Enter:0ź
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter:0J
rnn/TensorArray:05rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0ź
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/Enter:0Ć
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:0Irnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter:0
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Irnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter:0ş
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter:0
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0Rrnn/while/Enter:0Rrnn/while/Enter_1:0Rrnn/while/Enter_2:0Rrnn/while/Enter_3:0Rrnn/while/Enter_4:0Rrnn/while/Enter_5:0Rgradients/f_count_1:0"	
trainable_variables		
ő
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
ő
2rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const:0
b
softmax/Variable:0softmax/Variable/Assignsoftmax/Variable/read:02softmax/truncated_normal:0
]
softmax/Variable_1:0softmax/Variable_1/Assignsoftmax/Variable_1/read:02softmax/zeros:0"
	variablesôń
ő
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
ő
2rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform:0
ä
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const:0
b
softmax/Variable:0softmax/Variable/Assignsoftmax/Variable/read:02softmax/truncated_normal:0
]
softmax/Variable_1:0softmax/Variable_1/Assignsoftmax/Variable_1/read:02softmax/zeros:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros:0

9rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
ř
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros:0

7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0

7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros:0

9rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
ř
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros:0

7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0

softmax/Variable/Adam:0softmax/Variable/Adam/Assignsoftmax/Variable/Adam/read:02)softmax/Variable/Adam/Initializer/zeros:0

softmax/Variable/Adam_1:0softmax/Variable/Adam_1/Assignsoftmax/Variable/Adam_1/read:02+softmax/Variable/Adam_1/Initializer/zeros:0

softmax/Variable_1/Adam:0softmax/Variable_1/Adam/Assignsoftmax/Variable_1/Adam/read:02+softmax/Variable_1/Adam/Initializer/zeros:0

softmax/Variable_1/Adam_1:0 softmax/Variable_1/Adam_1/Assign softmax/Variable_1/Adam_1/read:02-softmax/Variable_1/Adam_1/Initializer/zeros:0O