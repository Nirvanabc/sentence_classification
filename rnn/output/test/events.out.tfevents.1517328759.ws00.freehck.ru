       �K"	  �]&��Abrain.Event:2R���B     ���	.[�]&��A"ۅ
W
inputsPlaceholder*
_output_shapes

:dd*
dtype0*
shape
:dd
X
targetsPlaceholder*
_output_shapes

:dd*
dtype0*
shape
:dd
N
	keep_probPlaceholder*
_output_shapes
:*
dtype0*
shape:
]
DropoutWrapperInit/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
_
DropoutWrapperInit/Const_1Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
_
DropoutWrapperInit_1/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
a
DropoutWrapperInit_1/Const_1Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
�
PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
KMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillKMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatPMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:d*
dtype0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
MMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillMMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
MMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFillMMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatRMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:d*
dtype0
�
TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
OMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV2NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1FillOMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:d*
dtype0
U
one_hot/on_valueConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
V
one_hot/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
P
one_hot/depthConst*
_output_shapes
: *
value
B :�*
dtype0
�
one_hotOneHotinputsone_hot/depthone_hot/on_valueone_hot/off_value*
TI0*#
_output_shapes
:dd�*
axis���������*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
M
range/startConst*
_output_shapes
: *
value	B :*
dtype0
M
range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
V
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
`
concat/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0rangeconcat/axis*

Tidx0*
_output_shapes
:*
N*
T0
b
	transpose	Transposeone_hotconcat*
Tperm0*#
_output_shapes
:dd�*
T0
^
	rnn/ShapeConst*
_output_shapes
:*!
valueB"d   d   �   *
dtype0
a
rnn/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
T0*
end_mask *
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
Index0
S
	rnn/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
U
rnn/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
Q
rnn/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
y

rnn/concatConcatV2	rnn/Constrnn/Const_1rnn/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
T
rnn/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
W
	rnn/zerosFill
rnn/concatrnn/zeros/Const*
_output_shapes

:dd*
T0
J
rnn/timeConst*
_output_shapes
: *
value	B : *
dtype0
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice*/
tensor_array_namernn/dynamic_rnn/output_0*
dtype0*
dynamic_size( *
element_shape:*
_output_shapes

:: *
clear_after_read(
�
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*.
tensor_array_namernn/dynamic_rnn/input_0*
dtype0*
dynamic_size( *
element_shape:*
_output_shapes

:: *
clear_after_read(
q
rnn/TensorArrayUnstack/ShapeConst*
_output_shapes
:*!
valueB"d   d   �   *
dtype0
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
T0*
end_mask *
new_axis_mask *
_output_shapes
: *
shrink_axis_mask*
Index0
d
"rnn/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
d
"rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
_output_shapes
: *
_class
loc:@transpose*
T0
�
rnn/while/EnterEnterrnn/time*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes
: *
T0*
parallel_iterations 
�
rnn/while/Enter_1Enterrnn/TensorArray:1*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes
: *
T0*
parallel_iterations 
�
rnn/while/Enter_2EnterJMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
rnn/while/Enter_3EnterLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
rnn/while/Enter_4EnterLMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
rnn/while/Enter_5EnterNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
_output_shapes
: : *
T0*
N
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
_output_shapes
: : *
T0*
N
|
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5* 
_output_shapes
:dd: *
T0*
N
�
rnn/while/Less/EnterEnterrnn/strided_slice*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations 
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
_output_shapes
: *
T0
F
rnn/while/LoopCondLoopCondrnn/while/Less*
_output_shapes
: 
�
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge*
T0
�
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1*
T0
�
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_2*
T0
�
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_3*
T0
�
rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_4*
T0
�
rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_5*
T0
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
�
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations 
�
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
_output_shapes
:	d�*
dtype0
�
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   �  *
dtype0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�ý*
dtype0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *��=*
dtype0
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *

seed *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
dtype0* 
_output_shapes
:
��
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��*
T0
�
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:�*
T0
�
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*'

frame_namernn/while/while_context*
is_constant(* 
_output_shapes
:
��*
T0*
parallel_iterations 
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
transpose_a( *
transpose_b( *
_output_shapes
:	d�*
T0
�
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes	
:�*
T0*
parallel_iterations 
�
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d�*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMulrnn/while/Identity_2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
_output_shapes

:dd*
T0
�
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
valueB"d   d   *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Hrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
seed2 *

seed *
_output_shapes

:dd*
T0*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
: *
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
�
:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes

:dd*
T0
�
5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/EnterEnter	keep_prob*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
_output_shapes
:*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
_output_shapes

:dd*
T0
�
Qrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB"�   �  *
dtype0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
 *��̽*
dtype0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
 *���=*
dtype0
�
Yrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *

seed *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
dtype0* 
_output_shapes
:
��
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
��*
T0
�
@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/ConstConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:�*
T0
�
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatConcatV2/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulrnn/while/Identity_5Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axis*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*'

frame_namernn/while/while_context*
is_constant(* 
_output_shapes
:
��*
T0*
parallel_iterations 
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter*
transpose_a( *
transpose_b( *
_output_shapes
:	d�*
T0
�
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes	
:�*
T0*
parallel_iterations 
�
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d�*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mulMulrnn/while/Identity_4Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split*
_output_shapes

:dd*
T0
�
?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
valueB"d   d   *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Hrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Shape*
seed2 *

seed *
_output_shapes

:dd*
T0*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes
: *
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
�
:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes

:dd*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform*
_output_shapes
:*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/add*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
_output_shapes

:dd*
T0
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*'

frame_namernn/while/while_context*
_output_shapes
:*
parallel_iterations 
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulrnn/while/Identity_1*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
f
rnn/while/add/yConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
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
�
rnn/while/NextIteration_2NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
rnn/while/NextIteration_3NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:dd*
T0
�
rnn/while/NextIteration_4NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
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
�
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes
: *"
_class
loc:@rnn/TensorArray
�
 rnn/TensorArrayStack/range/startConst*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
value	B : *
dtype0
�
 rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
value	B :*
dtype0
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:���������*"
_class
loc:@rnn/TensorArray
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
element_shape
:dd*"
_class
loc:@rnn/TensorArray*"
_output_shapes
:ddd*
dtype0
\
rnn/Const_2Const*
_output_shapes
:*
valueB"d   d   *
dtype0
U
rnn/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
J
rnn/RankConst*
_output_shapes
: *
value	B :*
dtype0
Q
rnn/range/startConst*
_output_shapes
: *
value	B :*
dtype0
Q
rnn/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0*
_output_shapes
:
f
rnn/concat_1/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
S
rnn/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
Tperm0*"
_output_shapes
:ddd*
T0
U
concat_1/concat_dimConst*
_output_shapes
: *
value	B :*
dtype0
P
concat_1Identityrnn/transpose*"
_output_shapes
:ddd*
T0
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����d   *
dtype0
c
ReshapeReshapeconcat_1Reshape/shape*
_output_shapes
:	�Nd*
Tshape0*
T0
o
softmax/truncated_normal/shapeConst*
_output_shapes
:*
valueB"d   �   *
dtype0
b
softmax/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
d
softmax/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
(softmax/truncated_normal/TruncatedNormalTruncatedNormalsoftmax/truncated_normal/shape*
seed2 *

seed *
_output_shapes
:	d�*
T0*
dtype0
�
softmax/truncated_normal/mulMul(softmax/truncated_normal/TruncatedNormalsoftmax/truncated_normal/stddev*
_output_shapes
:	d�*
T0
�
softmax/truncated_normalAddsoftmax/truncated_normal/mulsoftmax/truncated_normal/mean*
_output_shapes
:	d�*
T0
�
softmax/Variable
VariableV2*
	container *
shared_name *
_output_shapes
:	d�*
shape:	d�*
dtype0
�
softmax/Variable/AssignAssignsoftmax/Variablesoftmax/truncated_normal*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/readIdentitysoftmax/Variable*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
\
softmax/zerosConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
softmax/Variable_1
VariableV2*
	container *
shared_name *
_output_shapes	
:�*
shape:�*
dtype0
�
softmax/Variable_1/AssignAssignsoftmax/Variable_1softmax/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/readIdentitysoftmax/Variable_1*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
�
MatMulMatMulReshapesoftmax/Variable/read*
transpose_a( *
transpose_b( * 
_output_shapes
:
�N�*
T0
V
addAddMatMulsoftmax/Variable_1/read* 
_output_shapes
:
�N�*
T0
F
predictionsSoftmaxadd* 
_output_shapes
:
�N�*
T0
W
one_hot_1/on_valueConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
X
one_hot_1/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
R
one_hot_1/depthConst*
_output_shapes
: *
value
B :�*
dtype0
�
	one_hot_1OneHottargetsone_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
TI0*#
_output_shapes
:dd�*
axis���������*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
i
	Reshape_1Reshape	one_hot_1Reshape_1/shape* 
_output_shapes
:
�N�*
Tshape0*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
V
ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_1Const*
_output_shapes
:*
valueB"'  �   *
dtype0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_2Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*

axis *
N*
T0
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
d
concat_2/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
w
concat_2ConcatV2concat_2/values_0Sliceconcat_2/axis*

Tidx0*
_output_shapes
:*
N*
T0
\
	Reshape_2Reshapeaddconcat_2* 
_output_shapes
:
�N�*
Tshape0*
T0
H
Rank_3Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_2Const*
_output_shapes
:*
valueB"'  �   *
dtype0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_3Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*

axis *
N*
T0
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
concat_3/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_3/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_3ConcatV2concat_3/values_0Slice_1concat_3/axis*

Tidx0*
_output_shapes
:*
N*
T0
b
	Reshape_3Reshape	Reshape_1concat_3* 
_output_shapes
:
�N�*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*'
_output_shapes
:�N:
�N�*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_2SubRank_1Sub_2/y*
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
Slice_2/sizePackSub_2*
_output_shapes
:*

axis *
N*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
p
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
_output_shapes	
:�N*
Tshape0*
T0
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
\
MeanMean	Reshape_4Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
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
 *  �?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
S
gradients/f_countConst*
_output_shapes
: *
value	B : *
dtype0
�
gradients/f_count_1Entergradients/f_count*'

frame_namernn/while/while_context*
is_constant( *
_output_shapes
: *
T0*
parallel_iterations 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
_output_shapes
: : *
T0*
N
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
_output_shapes
: : *
T0
f
gradients/Add/yConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
�
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
gradients/b_countConst*
_output_shapes
: *
value	B :*
dtype0
�
gradients/b_count_1Entergradients/f_count_2*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes
: *
T0*
parallel_iterations 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
_output_shapes
: : *
T0*
N
�
gradients/GreaterEqual/EnterEntergradients/b_count*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
: *
T0*
parallel_iterations 
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
�
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
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
m
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
valueB:�N*
dtype0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:�N*
T0
d
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
valueB:�N*
dtype0
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*,
_class"
 loc:@gradients/Mean_grad/Shape*
valueB: *
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*

Tidx0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*,
_class"
 loc:@gradients/Mean_grad/Shape*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*

Tidx0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_output_shapes	
:�N*
T0
i
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
valueB:�N*
dtype0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
_output_shapes	
:�N*
Tshape0*
T0
m
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
�N�*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes
:	�N*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
�N�*
T0
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape* 
_output_shapes
:
�N�*
Tshape0*
T0
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape* 
_output_shapes
:
�N�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/Reshapesoftmax/Variable/read*
transpose_a( *
transpose_b(*
_output_shapes
:	�Nd*
T0
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_a(*
transpose_b( *
_output_shapes
:	d�*
T0
q
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*!
valueB"d   d   d   *
dtype0
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*"
_output_shapes
:ddd*
Tshape0*
T0
v
.gradients/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/concat_1*
_output_shapes
:*
T0
�
&gradients/rnn/transpose_grad/transpose	Transposegradients/Reshape_grad/Reshape.gradients/rnn/transpose_grad/InvertPermutation*
Tperm0*"
_output_shapes
:ddd*
T0
�
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes

:: *"
_class
loc:@rnn/TensorArray*
source	gradients
�
Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_1Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
T0
�
_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range&gradients/rnn/transpose_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
d
gradients/zerosConst*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_1Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_2Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_3Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
�
&gradients/rnn/while/Exit_1_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes
: *
T0*
parallel_iterations 
�
&gradients/rnn/while/Exit_2_grad/b_exitEntergradients/zeros*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_2*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
&gradients/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_3*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes

:dd*
T0*
parallel_iterations 
�
*gradients/rnn/while/Switch_1_grad/b_switchMerge&gradients/rnn/while/Exit_1_grad/b_exit1gradients/rnn/while/Switch_1_grad_1/NextIteration*
_output_shapes
: : *
T0*
N
�
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_5_grad/b_switchMerge&gradients/rnn/while/Exit_5_grad/b_exit1gradients/rnn/while/Switch_5_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
'gradients/rnn/while/Merge_1_grad/SwitchSwitch*gradients/rnn/while/Switch_1_grad/b_switchgradients/b_count_2*
_output_shapes
: : *=
_class3
1/loc:@gradients/rnn/while/Switch_1_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_5_grad/SwitchSwitch*gradients/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
T0
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
�
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
is_constant(*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:*
parallel_iterations 
�
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter)gradients/rnn/while/Merge_1_grad/Switch:1*
_output_shapes

:: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
source	gradients
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity)gradients/rnn/while/Merge_1_grad/Switch:1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
�
]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*
_output_shapes
: *%
_class
loc:@rnn/while/Identity*
valueB :
���������*
dtype0
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*
_output_shapes
:*

stack_name *%
_class
loc:@rnn/while/Identity*
	elem_type0
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity^gradients/Add*
_output_shapes
: *
swap_memory( *
T0
�
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0
�
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
�
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes

:dd*
dtype0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
out_type0*#
_output_shapes
:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*
	elem_type0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulMulNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
valueB :
���������*
dtype0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
	elem_type0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
_output_shapes
:*
Tshape0*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:���������:���������*
T0
�
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/EnterEnter	keep_prob*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*

stack_name *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
1gradients/rnn/while/Switch_1_grad_1/NextIterationNextIteration)gradients/rnn/while/Merge_1_grad/Switch:1*
_output_shapes
: *
T0
�
gradients/AddNAddN)gradients/rnn/while/Merge_5_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0
�
gradients/AddN_1AddN)gradients/rnn/while/Merge_4_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGrad*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
	elem_type0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_4*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_4*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_4^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
valueB *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*1

frame_name#!gradients/rnn/while/while_context*
is_constant(* 
_output_shapes
:
��*
T0*
parallel_iterations 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
_output_shapes
:	d�*
T0
�
hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*
	elem_type0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat^gradients/Add*
_output_shapes
:	d�*
swap_memory( *
T0
�
jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	d�*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
transpose_a(*
transpose_b( * 
_output_shapes
:
��*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes	
:�*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
T0*
N
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape*
_output_shapes

:dd*
T0*
Index0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes

:dd*
T0*
Index0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
��*
valueB
��*    *
dtype0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( * 
_output_shapes
:
��*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
T0*
N
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
out_type0*#
_output_shapes
:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
	elem_type0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
valueB :
���������*
dtype0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
	elem_type0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
_output_shapes
:*
Tshape0*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:���������:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*

stack_name *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
1gradients/rnn/while/Switch_5_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0
�
gradients/AddN_2AddN)gradients/rnn/while/Merge_3_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN_2^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN_2*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0
�
gradients/AddN_3AddN)gradients/rnn/while/Merge_2_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_3egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_3ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
	elem_type0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_2*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*

stack_name *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*

stack_name *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
valueB *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*1

frame_name#!gradients/rnn/while/while_context*
is_constant(* 
_output_shapes
:
��*
T0*
parallel_iterations 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
_output_shapes
:	d�*
T0
�
hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*

stack_name *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
	elem_type0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^gradients/Add*
_output_shapes
:	d�*
swap_memory( *
T0
�
jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
_output_shapes
:*
T0*
parallel_iterations 
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	d�*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
transpose_a(*
transpose_b( * 
_output_shapes
:
��*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
_output_shapes	
:�*
T0*
parallel_iterations 
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
T0*
N
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   �   *
dtype0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape*
_output_shapes
:	d�*
T0*
Index0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes

:dd*
T0*
Index0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
��*
valueB
��*    *
dtype0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( * 
_output_shapes
:
��*
T0*
parallel_iterations 
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
T0*
N
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0
�
global_norm/L2LossL2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_1L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_2L2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_3L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_4L2Lossgradients/MatMul_grad/MatMul_1*
_output_shapes
: *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
global_norm/L2Loss_5L2Lossgradients/add_grad/Reshape_1*
_output_shapes
: */
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5*
_output_shapes
:*

axis *
N*
T0
[
global_norm/ConstConst*
_output_shapes
:*
valueB: *
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
X
global_norm/Const_1Const*
_output_shapes
: *
valueB
 *   @*
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
clip_by_global_norm/truediv/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
valueB
 *  �@*
dtype0
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
valueB
 *  �@*
dtype0
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
�
clip_by_global_norm/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_2Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_3Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_4Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_5Mulgradients/MatMul_grad/MatMul_1clip_by_global_norm/mul*
_output_shapes
:	d�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	d�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
clip_by_global_norm/mul_6Mulgradients/add_grad/Reshape_1clip_by_global_norm/mul*
_output_shapes	
:�*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
_output_shapes	
:�*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
beta1_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB
 *fff?*
dtype0
�
beta1_power
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape: *
	container *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
beta1_power/readIdentitybeta1_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
beta2_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB
 *w�?*
dtype0
�
beta2_power
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape: *
	container *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
shape:
��*
	container *
dtype0* 
_output_shapes
:
��
�
>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Ernn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
8rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
�
'softmax/Variable/Adam/Initializer/zerosConst*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
valueB	d�*    *
dtype0
�
softmax/Variable/Adam
VariableV2*
shared_name *#
_class
loc:@softmax/Variable*
shape:	d�*
	container *
dtype0*
_output_shapes
:	d�
�
softmax/Variable/Adam/AssignAssignsoftmax/Variable/Adam'softmax/Variable/Adam/Initializer/zeros*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/Adam/readIdentitysoftmax/Variable/Adam*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
�
)softmax/Variable/Adam_1/Initializer/zerosConst*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
valueB	d�*    *
dtype0
�
softmax/Variable/Adam_1
VariableV2*
shared_name *#
_class
loc:@softmax/Variable*
shape:	d�*
	container *
dtype0*
_output_shapes
:	d�
�
softmax/Variable/Adam_1/AssignAssignsoftmax/Variable/Adam_1)softmax/Variable/Adam_1/Initializer/zeros*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/Adam_1/readIdentitysoftmax/Variable/Adam_1*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
�
)softmax/Variable_1/Adam/Initializer/zerosConst*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
valueB�*    *
dtype0
�
softmax/Variable_1/Adam
VariableV2*
shared_name *%
_class
loc:@softmax/Variable_1*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
softmax/Variable_1/Adam/AssignAssignsoftmax/Variable_1/Adam)softmax/Variable_1/Adam/Initializer/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/Adam/readIdentitysoftmax/Variable_1/Adam*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
�
+softmax/Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
valueB�*    *
dtype0
�
softmax/Variable_1/Adam_1
VariableV2*
shared_name *%
_class
loc:@softmax/Variable_1*
shape:�*
	container *
dtype0*
_output_shapes	
:�
�
 softmax/Variable_1/Adam_1/AssignAssignsoftmax/Variable_1/Adam_1+softmax/Variable_1/Adam_1/Initializer/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/Adam_1/readIdentitysoftmax/Variable_1/Adam_1*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o�:*
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
 *w�?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
FAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_nesterov( * 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
use_locking( 
�
DAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_nesterov( *
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
use_locking( 
�
FAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_nesterov( * 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
use_locking( 
�
DAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_nesterov( *
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
use_locking( 
�
&Adam/update_softmax/Variable/ApplyAdam	ApplyAdamsoftmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_nesterov( *
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
use_locking( 
�
(Adam/update_softmax/Variable_1/ApplyAdam	ApplyAdamsoftmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_nesterov( *
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking( 
�
AdamNoOpG^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*�
value�B�Bbeta1_powerBbeta2_powerB.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Bsoftmax/VariableBsoftmax/Variable/AdamBsoftmax/Variable/Adam_1Bsoftmax/Variable_1Bsoftmax/Variable_1/AdamBsoftmax/Variable_1/Adam_1*
dtype0
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
�
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
valueBBbeta1_power*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignbeta1_powersave/RestoreV2*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
q
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:* 
valueBBbeta2_power*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*C
value:B8B.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biassave/RestoreV2_2*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*H
value?B=B3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_3Assign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adamsave/RestoreV2_3*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1save/RestoreV2_4*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*E
value<B:B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelsave/RestoreV2_5* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adamsave/RestoreV2_6* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*L
valueCBAB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1save/RestoreV2_7* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*C
value:B8B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biassave/RestoreV2_8*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*H
value?B=B3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9Assign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adamsave/RestoreV2_9*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1save/RestoreV2_10*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*E
value<B:B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11Assign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelsave/RestoreV2_11* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adamsave/RestoreV2_12* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*L
valueCBAB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1save/RestoreV2_13* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*%
valueBBsoftmax/Variable*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assignsoftmax/Variablesave/RestoreV2_14*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
|
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:**
value!BBsoftmax/Variable/Adam*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assignsoftmax/Variable/Adamsave/RestoreV2_15*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
~
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*,
value#B!Bsoftmax/Variable/Adam_1*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignsoftmax/Variable/Adam_1save/RestoreV2_16*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
y
save/RestoreV2_17/tensor_namesConst*
_output_shapes
:*'
valueBBsoftmax/Variable_1*
dtype0
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_17Assignsoftmax/Variable_1save/RestoreV2_17*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
~
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*,
value#B!Bsoftmax/Variable_1/Adam*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_18Assignsoftmax/Variable_1/Adamsave/RestoreV2_18*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*.
value%B#Bsoftmax/Variable_1/Adam_1*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assignsoftmax/Variable_1/Adam_1save/RestoreV2_19*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19
�
initNoOp8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign8^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign^softmax/Variable/Assign^softmax/Variable_1/Assign^beta1_power/Assign^beta2_power/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign^softmax/Variable/Adam/Assign^softmax/Variable/Adam_1/Assign^softmax/Variable_1/Adam/Assign!^softmax/Variable_1/Adam_1/Assign"τ����     �|�H	N�^&��AJ��
�1�1
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
2	��
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	�
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
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
	elem_typetype�
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( �
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring �
�
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
element_shapeshape:�
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
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
Ttype*1.4.02v1.4.0-rc1-11-g130a514ۅ
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
DropoutWrapperInit/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
_
DropoutWrapperInit/Const_1Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
_
DropoutWrapperInit_1/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
a
DropoutWrapperInit_1/Const_1Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
�
PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
KMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
PMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
JMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillKMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatPMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:d*
dtype0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
MMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1FillMMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1RMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:d*
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
MMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/ConstNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_1RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
RMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
LMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zerosFillMMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concatRMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros/Const*
_output_shapes

:dd*
T0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5Const*
_output_shapes
:*
valueB:d*
dtype0
�
TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
OMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1ConcatV2NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_4NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_5TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1FillOMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/concat_1TMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes

:dd*
T0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_6Const*
_output_shapes
:*
valueB:d*
dtype0
�
NMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/Const_7Const*
_output_shapes
:*
valueB:d*
dtype0
U
one_hot/on_valueConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
V
one_hot/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
P
one_hot/depthConst*
_output_shapes
: *
value
B :�*
dtype0
�
one_hotOneHotinputsone_hot/depthone_hot/on_valueone_hot/off_value*
TI0*#
_output_shapes
:dd�*
axis���������*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
M
range/startConst*
_output_shapes
: *
value	B :*
dtype0
M
range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
V
rangeRangerange/startRankrange/delta*

Tidx0*
_output_shapes
:
`
concat/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0rangeconcat/axis*

Tidx0*
_output_shapes
:*
N*
T0
b
	transpose	Transposeone_hotconcat*
Tperm0*#
_output_shapes
:dd�*
T0
^
	rnn/ShapeConst*
_output_shapes
:*!
valueB"d   d   �   *
dtype0
a
rnn/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *
end_mask *
T0*

begin_mask *
new_axis_mask *
_output_shapes
: *
Index0
S
	rnn/ConstConst*
_output_shapes
:*
valueB:d*
dtype0
U
rnn/Const_1Const*
_output_shapes
:*
valueB:d*
dtype0
Q
rnn/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
y

rnn/concatConcatV2	rnn/Constrnn/Const_1rnn/concat/axis*

Tidx0*
_output_shapes
:*
N*
T0
T
rnn/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
W
	rnn/zerosFill
rnn/concatrnn/zeros/Const*
_output_shapes

:dd*
T0
J
rnn/timeConst*
_output_shapes
: *
value	B : *
dtype0
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice*
clear_after_read(*
dtype0*
dynamic_size( *
element_shape:*
_output_shapes

:: */
tensor_array_namernn/dynamic_rnn/output_0
�
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*
clear_after_read(*
dtype0*
dynamic_size( *
element_shape:*
_output_shapes

:: *.
tensor_array_namernn/dynamic_rnn/input_0
q
rnn/TensorArrayUnstack/ShapeConst*
_output_shapes
:*!
valueB"d   d   �   *
dtype0
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *
end_mask *
T0*

begin_mask *
new_axis_mask *
_output_shapes
: *
Index0
d
"rnn/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
d
"rnn/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
_output_shapes
: *
_class
loc:@transpose*
T0
�
rnn/while/EnterEnterrnn/time*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
�
rnn/while/Enter_1Enterrnn/TensorArray:1*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
�
rnn/while/Enter_2EnterJMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
rnn/while/Enter_3EnterLMultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
rnn/while/Enter_4EnterLMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
rnn/while/Enter_5EnterNMultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
_output_shapes
: : *
T0*
N
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
_output_shapes
: : *
T0*
N
|
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4* 
_output_shapes
:dd: *
T0*
N
|
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5* 
_output_shapes
:dd: *
T0*
N
�
rnn/while/Less/EnterEnterrnn/strided_slice*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
: *
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
�
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge*
T0
�
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1*
T0
�
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_2*
T0
�
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_3*
T0
�
rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_4*
T0
�
rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*(
_output_shapes
:dd:dd*$
_class
loc:@rnn/while/Merge_5*
T0
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
�
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
: *
T0
�
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
_output_shapes
:	d�*
dtype0
�
Qrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB"   �  *
dtype0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *�ý*
dtype0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
 *��=*
dtype0
�
Yrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *

seed *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
dtype0* 
_output_shapes
:
��
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Ornn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel* 
_output_shapes
:
��*
T0
�
@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/ConstConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
_output_shapes	
:�*
T0
�
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
transpose_a( *
transpose_b( *
_output_shapes
:	d�*
T0
�
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes	
:�*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d�*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMulrnn/while/Identity_2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
_output_shapes

:dd*
T0
�
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
valueB"d   d   *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Hrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
seed2 *

seed *
dtype0*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
: *
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
�
:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes

:dd*
T0
�
5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/EnterEnter	keep_prob*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
_output_shapes
:*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
_output_shapes

:dd*
T0
�
Qrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB"�   �  *
dtype0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
 *��̽*
dtype0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
 *���=*
dtype0
�
Yrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformQrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *

seed *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
dtype0* 
_output_shapes
:
��
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/subSubOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/maxOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Ornn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulYrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniformAddOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/mulOrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AssignAssign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelKrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel* 
_output_shapes
:
��*
T0
�
@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/ConstConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AssignAssign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
_output_shapes	
:�*
T0
�
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatConcatV2/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulrnn/while/Identity_5Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat/axis*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter*
transpose_a( *
transpose_b( *
_output_shapes
:	d�*
T0
�
Hrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes	
:�*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
_output_shapes
:	d�*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/ConstConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd*
	num_split*<
_output_shapes*
(:dd:dd:dd:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/yConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add/y*
_output_shapes

:dd*
T0
�
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mulMulrnn/while/Identity_4Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split*
_output_shapes

:dd*
T0
�
?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:1*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1*
_output_shapes

:dd*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split:3*
_output_shapes

:dd*
T0
�
@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
_output_shapes

:dd*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/ShapeConst^rnn/while/Identity*
_output_shapes
:*
valueB"d   d   *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/minConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *    *
dtype0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/maxConst^rnn/while/Identity*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
Hrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniformRandomUniform1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Shape*
seed2 *

seed *
dtype0*
_output_shapes

:dd*
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/subSub>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/max>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes
: *
T0
�
>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mulMulHrnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/RandomUniform>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/sub*
_output_shapes

:dd*
T0
�
:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniformAdd>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/mul>rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform/min*
_output_shapes

:dd*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform*
_output_shapes
:*
T0
�
1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/FloorFloor/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/add*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/divRealDiv@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_25rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter*
_output_shapes
:*
T0
�
/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulMul/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
_output_shapes

:dd*
T0
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*'

frame_namernn/while/while_context*
_output_shapes
:*
parallel_iterations 
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mulrnn/while/Identity_1*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
f
rnn/while/add/yConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
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
�
rnn/while/NextIteration_2NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
rnn/while/NextIteration_3NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
_output_shapes

:dd*
T0
�
rnn/while/NextIteration_4NextIteration@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1*
_output_shapes

:dd*
T0
�
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
�
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes
: *"
_class
loc:@rnn/TensorArray
�
 rnn/TensorArrayStack/range/startConst*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
value	B : *
dtype0
�
 rnn/TensorArrayStack/range/deltaConst*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
value	B :*
dtype0
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*#
_output_shapes
:���������*"
_class
loc:@rnn/TensorArray
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
element_shape
:dd*"
_class
loc:@rnn/TensorArray*"
_output_shapes
:ddd*
dtype0
\
rnn/Const_2Const*
_output_shapes
:*
valueB"d   d   *
dtype0
U
rnn/Const_3Const*
_output_shapes
:*
valueB:d*
dtype0
J
rnn/RankConst*
_output_shapes
: *
value	B :*
dtype0
Q
rnn/range/startConst*
_output_shapes
: *
value	B :*
dtype0
Q
rnn/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0*
_output_shapes
:
f
rnn/concat_1/values_0Const*
_output_shapes
:*
valueB"       *
dtype0
S
rnn/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
Tperm0*"
_output_shapes
:ddd*
T0
U
concat_1/concat_dimConst*
_output_shapes
: *
value	B :*
dtype0
P
concat_1Identityrnn/transpose*"
_output_shapes
:ddd*
T0
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����d   *
dtype0
c
ReshapeReshapeconcat_1Reshape/shape*
_output_shapes
:	�Nd*
Tshape0*
T0
o
softmax/truncated_normal/shapeConst*
_output_shapes
:*
valueB"d   �   *
dtype0
b
softmax/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
d
softmax/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *���=*
dtype0
�
(softmax/truncated_normal/TruncatedNormalTruncatedNormalsoftmax/truncated_normal/shape*
seed2 *

seed *
dtype0*
_output_shapes
:	d�*
T0
�
softmax/truncated_normal/mulMul(softmax/truncated_normal/TruncatedNormalsoftmax/truncated_normal/stddev*
_output_shapes
:	d�*
T0
�
softmax/truncated_normalAddsoftmax/truncated_normal/mulsoftmax/truncated_normal/mean*
_output_shapes
:	d�*
T0
�
softmax/Variable
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes
:	d�*
shape:	d�
�
softmax/Variable/AssignAssignsoftmax/Variablesoftmax/truncated_normal*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/readIdentitysoftmax/Variable*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
\
softmax/zerosConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
softmax/Variable_1
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:�*
shape:�
�
softmax/Variable_1/AssignAssignsoftmax/Variable_1softmax/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/readIdentitysoftmax/Variable_1*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
�
MatMulMatMulReshapesoftmax/Variable/read*
transpose_a( *
transpose_b( * 
_output_shapes
:
�N�*
T0
V
addAddMatMulsoftmax/Variable_1/read* 
_output_shapes
:
�N�*
T0
F
predictionsSoftmaxadd* 
_output_shapes
:
�N�*
T0
W
one_hot_1/on_valueConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
X
one_hot_1/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
R
one_hot_1/depthConst*
_output_shapes
: *
value
B :�*
dtype0
�
	one_hot_1OneHottargetsone_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
TI0*#
_output_shapes
:dd�*
axis���������*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
i
	Reshape_1Reshape	one_hot_1Reshape_1/shape* 
_output_shapes
:
�N�*
Tshape0*
T0
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
V
ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_1Const*
_output_shapes
:*
valueB"'  �   *
dtype0
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_2Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*

axis *
N*
T0
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
d
concat_2/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
w
concat_2ConcatV2concat_2/values_0Sliceconcat_2/axis*

Tidx0*
_output_shapes
:*
N*
T0
\
	Reshape_2Reshapeaddconcat_2* 
_output_shapes
:
�N�*
Tshape0*
T0
H
Rank_3Const*
_output_shapes
: *
value	B :*
dtype0
X
Shape_2Const*
_output_shapes
:*
valueB"'  �   *
dtype0
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_3Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*

axis *
N*
T0
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
concat_3/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
O
concat_3/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_3ConcatV2concat_3/values_0Slice_1concat_3/axis*

Tidx0*
_output_shapes
:*
N*
T0
b
	Reshape_3Reshape	Reshape_1concat_3* 
_output_shapes
:
�N�*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_2	Reshape_3*'
_output_shapes
:�N:
�N�*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_2SubRank_1Sub_2/y*
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
Slice_2/sizePackSub_2*
_output_shapes
:*

axis *
N*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
p
	Reshape_4ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
_output_shapes	
:�N*
Tshape0*
T0
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
\
MeanMean	Reshape_4Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
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
 *  �?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
S
gradients/f_countConst*
_output_shapes
: *
value	B : *
dtype0
�
gradients/f_count_1Entergradients/f_count*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
_output_shapes
: : *
T0*
N
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
_output_shapes
: : *
T0
f
gradients/Add/yConst^rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
_output_shapes
: *
T0
�
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
gradients/b_countConst*
_output_shapes
: *
value	B :*
dtype0
�
gradients/b_count_1Entergradients/f_count_2*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
_output_shapes
: : *
T0*
N
�
gradients/GreaterEqual/EnterEntergradients/b_count*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
: *
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
�
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
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
m
"gradients/Mean_grad/Tile/multiplesConst*
_output_shapes
:*
valueB:�N*
dtype0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshape"gradients/Mean_grad/Tile/multiples*

Tmultiples0*
_output_shapes	
:�N*
T0
d
gradients/Mean_grad/ShapeConst*
_output_shapes
:*
valueB:�N*
dtype0
^
gradients/Mean_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*
_output_shapes
:*,
_class"
 loc:@gradients/Mean_grad/Shape*
valueB: *
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shapegradients/Mean_grad/Const*

Tidx0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
	keep_dims( 
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*,
_class"
 loc:@gradients/Mean_grad/Shape*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_1gradients/Mean_grad/Const_1*

Tidx0*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *,
_class"
 loc:@gradients/Mean_grad/Shape*
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_output_shapes	
:�N*
T0
i
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
valueB:�N*
dtype0
�
 gradients/Reshape_4_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_4_grad/Shape*
_output_shapes	
:�N*
Tshape0*
T0
m
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
�N�*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_4_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
_output_shapes
:	�N*
T0*

Tdim0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1* 
_output_shapes
:
�N�*
T0
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
�
 gradients/Reshape_2_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_2_grad/Shape* 
_output_shapes
:
�N�*
Tshape0*
T0
i
gradients/add_grad/ShapeConst*
_output_shapes
:*
valueB"'  �   *
dtype0
e
gradients/add_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSum gradients/Reshape_2_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape* 
_output_shapes
:
�N�*
Tshape0*
T0
�
gradients/add_grad/Sum_1Sum gradients/Reshape_2_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes	
:�*
Tshape0*
T0
�
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/Reshapesoftmax/Variable/read*
transpose_a( *
transpose_b(*
_output_shapes
:	�Nd*
T0
�
gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
transpose_a(*
transpose_b( *
_output_shapes
:	d�*
T0
q
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*!
valueB"d   d   d   *
dtype0
�
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*"
_output_shapes
:ddd*
Tshape0*
T0
v
.gradients/rnn/transpose_grad/InvertPermutationInvertPermutationrnn/concat_1*
_output_shapes
:*
T0
�
&gradients/rnn/transpose_grad/transpose	Transposegradients/Reshape_grad/Reshape.gradients/rnn/transpose_grad/InvertPermutation*
Tperm0*"
_output_shapes
:ddd*
T0
�
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_1*
_output_shapes

:: *"
_class
loc:@rnn/TensorArray*
source	gradients
�
Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_1Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *"
_class
loc:@rnn/TensorArray*
T0
�
_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range&gradients/rnn/transpose_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
d
gradients/zerosConst*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_1Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_2Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
f
gradients/zeros_3Const*
_output_shapes

:dd*
valueBdd*    *
dtype0
�
&gradients/rnn/while/Exit_1_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes
: *
T0
�
&gradients/rnn/while/Exit_2_grad/b_exitEntergradients/zeros*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_2*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
&gradients/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_3*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes

:dd*
T0
�
*gradients/rnn/while/Switch_1_grad/b_switchMerge&gradients/rnn/while/Exit_1_grad/b_exit1gradients/rnn/while/Switch_1_grad_1/NextIteration*
_output_shapes
: : *
T0*
N
�
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
*gradients/rnn/while/Switch_5_grad/b_switchMerge&gradients/rnn/while/Exit_5_grad/b_exit1gradients/rnn/while/Switch_5_grad_1/NextIteration* 
_output_shapes
:dd: *
T0*
N
�
'gradients/rnn/while/Merge_1_grad/SwitchSwitch*gradients/rnn/while/Switch_1_grad/b_switchgradients/b_count_2*
_output_shapes
: : *=
_class3
1/loc:@gradients/rnn/while/Switch_1_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
T0
�
'gradients/rnn/while/Merge_5_grad/SwitchSwitch*gradients/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*(
_output_shapes
:dd:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
T0
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
�
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
is_constant(*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:*
parallel_iterations 
�
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter)gradients/rnn/while/Merge_1_grad/Switch:1*
_output_shapes

:: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
source	gradients
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity)gradients/rnn/while/Merge_1_grad/Switch:1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul*
T0
�
]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_sizeConst*
_output_shapes
: *%
_class
loc:@rnn/while/Identity*
valueB :
���������*
dtype0
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2]gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc/max_size*
_output_shapes
:*%
_class
loc:@rnn/while/Identity*

stack_name *
	elem_type0
�
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity^gradients/Add*
_output_shapes
: *
swap_memory( *
T0
�
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
_output_shapes
: *
	elem_type0
�
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2`^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2N^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2]^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2_^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2a^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2e^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
�
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes

:dd*
dtype0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
out_type0*#
_output_shapes
:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape*

stack_name *
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1*

stack_name *
	elem_type0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor*

stack_name *
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_1/dropout/Floor^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulMulNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*
valueB :
���������*
dtype0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div*

stack_name *
	elem_type0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/StackPopV2Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
_output_shapes
:*
Tshape0*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1*

stack_name *
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:���������:���������*
T0
�
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/EnterEnter	keep_prob*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2*

stack_name *
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
1gradients/rnn/while/Switch_1_grad_1/NextIterationNextIteration)gradients/rnn/while/Merge_1_grad/Switch:1*
_output_shapes
: *
T0
�
gradients/AddNAddN)gradients/rnn/while/Merge_5_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Reshape*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1*

stack_name *
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0
�
gradients/AddN_1AddN)gradients/rnn/while/Merge_4_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_1_grad/TanhGrad*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid*

stack_name *
	elem_type0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_4*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*'
_class
loc:@rnn/while/Identity_4*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_4^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1*

stack_name *
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
valueB *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
_output_shapes
:	d�*
T0
�
hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat*

stack_name *
	elem_type0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat^gradients/Add*
_output_shapes
:	d�*
swap_memory( *
T0
�
jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	d�*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/split_grad/concat*
transpose_a(*
transpose_b( * 
_output_shapes
:
��*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
T0*
N
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape*
_output_shapes

:dd*
T0*
Index0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes

:dd*
T0*
Index0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
��*
valueB
��*    *
dtype0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations * 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
T0*
N
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ShapeShape/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
out_type0*#
_output_shapes
:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1Shape1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape*

stack_name *
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*
valueB :
���������*
dtype0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1/max_size*
_output_shapes
:*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1*

stack_name *
	elem_type0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc/max_size*
_output_shapes
:*D
_class:
86loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*

stack_name *
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter1rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/SliceMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/StackPopV2*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/SumSumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mulTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*
valueB :
���������*
dtype0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_accStackV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*B
_class8
64loc:@rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div*

stack_name *
	elem_type0
�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:*
	elem_type0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1MulOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/Sum_1agradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
_output_shapes
:*
Tshape0*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1Shape5rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter^rnn/while/Identity*
out_type0*#
_output_shapes
:���������*
T0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_sizeConst*
_output_shapes
: *Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*
valueB :
���������*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_accStackV2cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc/max_size*
_output_shapes
:*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1*

stack_name *
	elem_type0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/EnterFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_1^gradients/Add*#
_output_shapes
:���������*
swap_memory( *
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*#
_output_shapes
:���������*
	elem_type0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*2
_output_shapes 
:���������:���������*
T0
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivRealDivFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumSumFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDivTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/ReshapeReshapeBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/SumDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_sizeConst*
_output_shapes
: *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
valueB :
���������*
dtype0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_accStackV2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc/max_size*
_output_shapes
:*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*

stack_name *
	elem_type0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegNegMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/StackPopV2*
_output_shapes

:dd*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1RealDivBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/NegLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2RealDivHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/RealDiv/Enter*
_output_shapes
:*
T0
�
Bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulMulFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/ReshapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/RealDiv_2*
_output_shapes
:*
T0
�
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1SumBgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/mulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape_1ReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Sum_1_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
_output_shapes
:*
Tshape0*
T0
�
1gradients/rnn/while/Switch_5_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0
�
gradients/AddN_2AddN)gradients/rnn/while/Merge_3_grad/Switch:1Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Reshape*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulMulgradients/AddN_2^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc/max_size*
_output_shapes
:*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1*

stack_name *
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2gradients/AddN_2*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/Reshape_1*
_output_shapes

:dd*
T0
�
gradients/AddN_3AddN)gradients/rnn/while/Merge_2_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1_grad/TanhGrad*
_output_shapes

:dd*=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch*
N*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumSumgradients/AddN_3egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_3ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_accStackV2`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc/max_size*
_output_shapes
:*U
_classK
IGloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*

stack_name *
	elem_type0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPushV2StackPushV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/EnterBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/EnterEnterWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2
StackPopV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSumQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mulcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_2*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc/max_size*
_output_shapes
:*'
_class
loc:@rnn/while/Identity_2*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enterrnn/while/Identity_2^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ShapeWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_sizeConst*
_output_shapes
: *R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_accStackV2bgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc/max_size*
_output_shapes
:*R
_classH
FDloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*

stack_name *
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPushV2StackPushV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/EnterEnterYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2
StackPopV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulMulYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumSumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mulegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/ReshapeReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_sizeConst*
_output_shapes
: *W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_accStackV2dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc/max_size*
_output_shapes
:*W
_classM
KIloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1*

stack_name *
	elem_type0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPushV2StackPushV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/EnterDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1^gradients/Add*
_output_shapes

:dd*
swap_memory( *
T0
�
fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/EnterEnter[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2
StackPopV2fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes

:dd*
	elem_type0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1Mul`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1SumUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1ReshapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Sum_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Shape_1*
_output_shapes

:dd*
Tshape0*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape_1*
_output_shapes

:dd*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape*
_output_shapes

:dd*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradTanhGrad^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/Reshape_1*
_output_shapes

:dd*
T0
�
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/Reshape*
_output_shapes

:dd*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1Const^gradients/Sub*
_output_shapes
: *
valueB *
dtype0
�
cgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ShapeUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradcgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/ReshapeReshapeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/SumSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape*
_output_shapes

:dd*
Tshape0*
T0
�
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Sum]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_grad/SigmoidGradegradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_1ReshapeSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Sum_1Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concatConcatV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_grad/TanhGradUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_grad/Reshape_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
_output_shapes
:	d�*
N*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*
T0
�
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMulMatMulVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_a( *
transpose_b(*
_output_shapes
:	d�*
T0
�
hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_sizeConst*
_output_shapes
: *T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc/max_size*
_output_shapes
:*T
_classJ
HFloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat*

stack_name *
	elem_type0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*'

frame_namernn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/EnterArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat^gradients/Add*
_output_shapes
:	d�*
swap_memory( *
T0
�
jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnter_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant(*
parallel_iterations *
_output_shapes
:*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
_output_shapes
:	d�*
	elem_type0
�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1MatMuldgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_grad/concat*
transpose_a(*
transpose_b( * 
_output_shapes
:
��*
T0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
_output_shapes	
:�*
valueB�*    *
dtype0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Enter]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations *
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2Merge_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
_output_shapes
	:�: *
T0*
N
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitch_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*"
_output_shapes
:�:�*
T0
�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/AddAdd`gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
egradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:�*
T0
�
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3Exit^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:�*
T0
�
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstConst^gradients/Sub*
_output_shapes
: *
value	B :*
dtype0
�
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modFloorModZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/mod/ConstUgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeConst^gradients/Sub*
_output_shapes
:*
valueB"d   �   *
dtype0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*
valueB"d   d   *
dtype0
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffsetTgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/modVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ShapeXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1* 
_output_shapes
::*
N
�
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/SliceSliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffsetVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape*
_output_shapes
:	d�*
T0*
Index0
�
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1SliceWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/ConcatOffset:1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Shape_1*
_output_shapes

:dd*
T0*
Index0
�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
��*
valueB
��*    *
dtype0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc*1

frame_name#!gradients/rnn/while/while_context*
is_constant( *
parallel_iterations * 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2Merge^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_1dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIteration*"
_output_shapes
:
��: *
T0*
N
�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitch^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_output_shapes
:
��:
��*
T0
�
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/AddAdd_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIterationZgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
��*
T0
�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3Exit]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
��*
T0
�
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIterationXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_grad/Slice_1*
_output_shapes

:dd*
T0
�
global_norm/L2LossL2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_1L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_2L2Loss^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
_output_shapes
: *q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_3L2Loss_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
_output_shapes
: *r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
global_norm/L2Loss_4L2Lossgradients/MatMul_grad/MatMul_1*
_output_shapes
: *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
global_norm/L2Loss_5L2Lossgradients/add_grad/Reshape_1*
_output_shapes
: */
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5*
_output_shapes
:*

axis *
N*
T0
[
global_norm/ConstConst*
_output_shapes
:*
valueB: *
dtype0
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
X
global_norm/Const_1Const*
_output_shapes
: *
valueB
 *   @*
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
clip_by_global_norm/truediv/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
_output_shapes
: *
T0
^
clip_by_global_norm/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
valueB
 *  �@*
dtype0
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
_output_shapes
: *
T0
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
_output_shapes
: *
T0
^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
valueB
 *  �@*
dtype0
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
�
clip_by_global_norm/mul_1Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_2Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_3Mul^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3* 
_output_shapes
:
��*q
_classg
ecloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_4Mul_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
_output_shapes	
:�*r
_classh
fdloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
�
clip_by_global_norm/mul_5Mulgradients/MatMul_grad/MatMul_1clip_by_global_norm/mul*
_output_shapes
:	d�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	d�*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
�
clip_by_global_norm/mul_6Mulgradients/add_grad/Reshape_1clip_by_global_norm/mul*
_output_shapes	
:�*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
_output_shapes	
:�*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
�
beta1_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB
 *fff?*
dtype0
�
beta1_power
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0*
	container *
shape: *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
beta1_power/readIdentitybeta1_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
beta2_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB
 *w�?*
dtype0
�
beta2_power
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0*
	container *
shape: *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
�
Ernn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
8rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamGrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zerosConst* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
valueB
��*    *
dtype0
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0*
	container *
shape:
��* 
_output_shapes
:
��
�
>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/AssignAssign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/readIdentity7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0
�
Ernn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/AssignAssign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamErnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
8rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/readIdentity3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
�
Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/AssignAssign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/readIdentity5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0
�
'softmax/Variable/Adam/Initializer/zerosConst*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
valueB	d�*    *
dtype0
�
softmax/Variable/Adam
VariableV2*
shared_name *#
_class
loc:@softmax/Variable*
dtype0*
	container *
shape:	d�*
_output_shapes
:	d�
�
softmax/Variable/Adam/AssignAssignsoftmax/Variable/Adam'softmax/Variable/Adam/Initializer/zeros*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/Adam/readIdentitysoftmax/Variable/Adam*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
�
)softmax/Variable/Adam_1/Initializer/zerosConst*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
valueB	d�*    *
dtype0
�
softmax/Variable/Adam_1
VariableV2*
shared_name *#
_class
loc:@softmax/Variable*
dtype0*
	container *
shape:	d�*
_output_shapes
:	d�
�
softmax/Variable/Adam_1/AssignAssignsoftmax/Variable/Adam_1)softmax/Variable/Adam_1/Initializer/zeros*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
�
softmax/Variable/Adam_1/readIdentitysoftmax/Variable/Adam_1*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0
�
)softmax/Variable_1/Adam/Initializer/zerosConst*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
valueB�*    *
dtype0
�
softmax/Variable_1/Adam
VariableV2*
shared_name *%
_class
loc:@softmax/Variable_1*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
softmax/Variable_1/Adam/AssignAssignsoftmax/Variable_1/Adam)softmax/Variable_1/Adam/Initializer/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/Adam/readIdentitysoftmax/Variable_1/Adam*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
�
+softmax/Variable_1/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
valueB�*    *
dtype0
�
softmax/Variable_1/Adam_1
VariableV2*
shared_name *%
_class
loc:@softmax/Variable_1*
dtype0*
	container *
shape:�*
_output_shapes	
:�
�
 softmax/Variable_1/Adam_1/AssignAssignsoftmax/Variable_1/Adam_1+softmax/Variable_1/Adam_1/Initializer/zeros*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
softmax/Variable_1/Adam_1/readIdentitysoftmax/Variable_1/Adam_1*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o�:*
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
 *w�?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
FAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_nesterov( * 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
use_locking( 
�
DAdam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_nesterov( *
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
use_locking( 
�
FAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdam	ApplyAdam0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_nesterov( * 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
use_locking( 
�
DAdam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam	ApplyAdam.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
use_nesterov( *
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
use_locking( 
�
&Adam/update_softmax/Variable/ApplyAdam	ApplyAdamsoftmax/Variablesoftmax/Variable/Adamsoftmax/Variable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_nesterov( *
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
use_locking( 
�
(Adam/update_softmax/Variable_1/ApplyAdam	ApplyAdamsoftmax/Variable_1softmax/Variable_1/Adamsoftmax/Variable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_nesterov( *
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2G^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking( 
�
AdamNoOpG^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/ApplyAdamG^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/ApplyAdamE^Adam/update_rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/ApplyAdam'^Adam/update_softmax/Variable/ApplyAdam)^Adam/update_softmax/Variable_1/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*�
value�B�Bbeta1_powerBbeta2_powerB.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biasB3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/AdamB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelB5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/AdamB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1Bsoftmax/VariableBsoftmax/Variable/AdamBsoftmax/Variable/Adam_1Bsoftmax/Variable_1Bsoftmax/Variable_1/AdamBsoftmax/Variable_1/Adam_1*
dtype0
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
�
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
valueBBbeta1_power*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/AssignAssignbeta1_powersave/RestoreV2*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
q
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:* 
valueBBbeta2_power*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
_output_shapes
: *A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*C
value:B8B.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assign.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biassave/RestoreV2_2*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*H
value?B=B3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_3Assign3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adamsave/RestoreV2_3*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1save/RestoreV2_4*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*E
value<B:B0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assign0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelsave/RestoreV2_5* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adamsave/RestoreV2_6* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*L
valueCBAB7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1save/RestoreV2_7* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*C
value:B8B.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_8Assign.rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biassave/RestoreV2_8*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*H
value?B=B3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_9Assign3rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adamsave/RestoreV2_9*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1save/RestoreV2_10*
_output_shapes	
:�*A
_class7
53loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*E
value<B:B0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11Assign0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernelsave/RestoreV2_11* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*J
valueAB?B5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adamsave/RestoreV2_12* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*L
valueCBAB7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1save/RestoreV2_13* 
_output_shapes
:
��*C
_class9
75loc:@rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel*
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*%
valueBBsoftmax/Variable*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_14Assignsoftmax/Variablesave/RestoreV2_14*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
|
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:**
value!BBsoftmax/Variable/Adam*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assignsoftmax/Variable/Adamsave/RestoreV2_15*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
~
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*,
value#B!Bsoftmax/Variable/Adam_1*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignsoftmax/Variable/Adam_1save/RestoreV2_16*
_output_shapes
:	d�*#
_class
loc:@softmax/Variable*
T0*
validate_shape(*
use_locking(
y
save/RestoreV2_17/tensor_namesConst*
_output_shapes
:*'
valueBBsoftmax/Variable_1*
dtype0
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_17Assignsoftmax/Variable_1save/RestoreV2_17*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
~
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*,
value#B!Bsoftmax/Variable_1/Adam*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_18Assignsoftmax/Variable_1/Adamsave/RestoreV2_18*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*.
value%B#Bsoftmax/Variable_1/Adam_1*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assignsoftmax/Variable_1/Adam_1save/RestoreV2_19*
_output_shapes	
:�*%
_class
loc:@softmax/Variable_1*
T0*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19
�
initNoOp8^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign8^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign6^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign^softmax/Variable/Assign^softmax/Variable_1/Assign^beta1_power/Assign^beta2_power/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign?^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign;^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign=^rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign^softmax/Variable/Adam/Assign^softmax/Variable/Adam_1/Assign^softmax/Variable_1/Adam/Assign!^softmax/Variable_1/Adam_1/Assign""ܡ
while_contextɡš
��
rnn/while/while_context *rnn/while/LoopCond:02rnn/while/Merge:0:rnn/while/Identity:0Brnn/while/Exit:0Brnn/while/Exit_1:0Brnn/while/Exit_2:0Brnn/while/Exit_3:0Brnn/while/Exit_4:0Brnn/while/Exit_5:0Bgradients/f_count_2:0J��
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
<rnn/while/rnn/multi_rnn_cell/cell_1/dropout/random_uniform:0�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul/Enter:0:
rnn/TensorArray_1:0#rnn/while/TensorArrayReadV3/Enter:0�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter:0J
rnn/TensorArray:05rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul_1/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul_1/Enter:0-
rnn/strided_slice:0rnn/while/Less/Enter:0�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:0Irnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul/Enter:0�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:0Jrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/BiasAdd/Enter:0�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_1_grad/mul_1/Enter:0�
agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0agradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/f_acc:0Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_grad/mul/Enter:0�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter:0�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/Neg/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul/Enter:0�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/BroadcastGradientArgs/Enter:0�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:0Irnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter:0i
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%rnn/while/TensorArrayReadV3/Enter_1:0�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul_1/Enter:0�
\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/div_grad/BroadcastGradientArgs/Enter:0�
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/mul_1/Enter:0�
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/f_acc:0Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_grad/mul/Enter:0�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1_grad/mul_1/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul/Enter:0�
^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0�
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/f_acc:0]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2_grad/mul_1/Enter:0�
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul/Enter:0�
agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc:0agradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/MatMul_grad/MatMul_1/Enter:0�
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/f_acc:0[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/basic_lstm_cell/mul_2_grad/mul/Enter:0F
keep_prob:07rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add/Enter:0�
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/f_acc:0Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/dropout/div_grad/Neg/Enter:0�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:0Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter:0�
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul_grad/mul_1/Enter:0Rrnn/while/Enter:0Rrnn/while/Enter_1:0Rrnn/while/Enter_2:0Rrnn/while/Enter_3:0Rrnn/while/Enter_4:0Rrnn/while/Enter_5:0Rgradients/f_count_1:0"�	
trainable_variables�	�	
�
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
�
2rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const:0
b
softmax/Variable:0softmax/Variable/Assignsoftmax/Variable/read:02softmax/truncated_normal:0
]
softmax/Variable_1:0softmax/Variable_1/Assignsoftmax/Variable_1/read:02softmax/zeros:0"�
	variables��
�
2rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Initializer/Const:0
�
2rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:07rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Assign7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/read:02Mrnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Initializer/random_uniform:0
�
0rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:05rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Assign5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/read:02Brnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Initializer/Const:0
b
softmax/Variable:0softmax/Variable/Assignsoftmax/Variable/read:02softmax/truncated_normal:0
]
softmax/Variable_1:0softmax/Variable_1/Assignsoftmax/Variable_1/read:02softmax/zeros:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
�
9rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
�
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam/Initializer/zeros:0
�
7rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam:0<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Assign<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/read:02Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
�
9rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1:0>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Assign>rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/read:02Krnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
�
5rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam:0:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Assign:rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/read:02Grnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam/Initializer/zeros:0
�
7rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1:0<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Assign<rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/read:02Irnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
�
softmax/Variable/Adam:0softmax/Variable/Adam/Assignsoftmax/Variable/Adam/read:02)softmax/Variable/Adam/Initializer/zeros:0
�
softmax/Variable/Adam_1:0softmax/Variable/Adam_1/Assignsoftmax/Variable/Adam_1/read:02+softmax/Variable/Adam_1/Initializer/zeros:0
�
softmax/Variable_1/Adam:0softmax/Variable_1/Adam/Assignsoftmax/Variable_1/Adam/read:02+softmax/Variable_1/Adam/Initializer/zeros:0
�
softmax/Variable_1/Adam_1:0 softmax/Variable_1/Adam_1/Assign softmax/Variable_1/Adam_1/read:02-softmax/Variable_1/Adam_1/Initializer/zeros:0"
train_op

Adam2�K�        )��P	Յ&^&��A*


batch_loss,��@��-�        )��P	G7^&��A*


batch_lossKB�@�C�        )��P	��G^&��A*


batch_loss Ԡ@V9$$        )��P	�ew^&��A*


batch_loss,�@��y�        )��P	��^&��A*


batch_loss=4�@Ȩ��        )��P	>�^&��A*


batch_loss�g�@A���        )��P	�6�^&��A*


batch_loss65�@�V��        )��P	�z�^&��A*


batch_loss�A�@i�V�        )��P	�)�^&��A	*


batch_loss�,�@O�W�        )��P	�_&��A
*


batch_loss�f�@`���        )��P	�z!_&��A*


batch_loss��@x榱        )��P	{3_&��A*


batch_loss��@�_��        )��P	 X`_&��A*


batch_loss]��@|��(        )��P	� q_&��A*


batch_loss}@<���        )��P	[��_&��A*


batch_loss4=z@��@]        )��P	�_&��A*


batch_loss(�u@�B        )��P	���_&��A*


batch_loss��q@T2�5        )��P	�8�_&��A*


batch_loss^/o@2K�        )��P	�0`&��A*


batch_loss�wm@�3�        )��P	�`&��A*


batch_loss40k@ȭk�        )��P	\�$`&��A*


batch_loss�i@V��=        )��P	q�R`&��A*


batch_loss��f@u�M        )��P	�md`&��A*


batch_loss��f@�c�{        )��P	��u`&��A*


batch_lossS�c@0oi        )��P	��`&��A*


batch_loss[�c@ ��8        )��P	�=�`&��A*


batch_loss�Dc@RG[        )��P	�!�`&��A*


batch_loss�	b@�b͹        )��P	���`&��A*


batch_lossT�b@��$�        )��P	)�a&��A*


batch_loss=va@F�؝        )��P	�a&��A*


batch_loss��`@!�        )��P	��Ga&��A*


batch_lossu7a@�*D�        )��P	�GXa&��A *


batch_loss`_@t<�L        )��P	`�ia&��A!*


batch_lossT�^@�g,1        )��P	���a&��A"*


batch_loss�t^@���_        )��P	G�a&��A#*


batch_lossc�]@	q4i        )��P	`��a&��A$*


batch_lossd�\@�Qp�        )��P	nK�a&��A%*


batch_loss��\@�7x        )��P	X|b&��A&*


batch_loss_\@.�        )��P	�Jb&��A'*


batch_loss��]@���]        )��P	��Ab&��A(*


batch_loss��]@�G3        )��P	��Qb&��A)*


batch_loss9�\@�	+        )��P	�Pcb&��A**


batch_loss4�\@�:G        )��P	Ů�b&��A+*


batch_loss�[@Z�*�        )��P	�H�b&��A,*


batch_loss��Z@x�"        )��P	az�b&��A-*


batch_loss&,\@�EJ:        )��P	Y��b&��A.*


batch_loss8W\@ ¢�        )��P	
�b&��A/*


batch_loss 6[@��1        )��P	��c&��A0*


batch_loss[�Z@"�        )��P	D>c&��A1*


batch_lossA�Y@R�B�        )��P	D�Mc&��A2*


batch_loss�;Y@7���        )��P	�W_c&��A3*


batch_loss7Y@ykV}        )��P	_��c&��A4*


batch_loss��Z@��5�        )��P	Kd�c&��A5*


batch_loss��X@%V        )��P	I̱c&��A6*


batch_loss��X@}��        )��P	���c&��A7*


batch_lossqeX@�a7�        )��P	�C�c&��A8*


batch_lossu�W@'Sr        )��P	�d&��A9*


batch_loss'�X@��]        )��P	:�5d&��A:*


batch_loss��X@H��F        )��P	VeFd&��A;*


batch_loss�aX@`<�t        )��P	�LWd&��A<*


batch_lossi�Y@� ]        )��P	zQ�d&��A=*


batch_loss��W@uFI�        )��P	�H�d&��A>*


batch_lossxLX@�D        )��P	>_�d&��A?*


batch_loss�mX@d6�        )��P	���d&��A@*


batch_loss?MX@}v��        )��P	t��d&��AA*


batch_loss��V@�^7        )��P	4��d&��AB*


batch_loss	W@��Ы        )��P	��*e&��AC*


batch_loss�{W@}^�        )��P	�<e&��AD*


batch_loss��V@)�Υ        )��P	��Me&��AE*


batch_losshW@i��s        )��P	�e&��AF*


batch_loss��X@��S+        )��P	I�e&��AG*


batch_loss�W@y�n        )��P	�i�e&��AH*


batch_loss�EX@5���        )��P	1��e&��AI*


batch_lossؑV@�t2\        )��P	���e&��AJ*


batch_lossrmV@B��        )��P	(h�e&��AK*


batch_loss�~V@6��(        )��P	�%f&��AL*


batch_loss��V@��4�        )��P	��7f&��AM*


batch_loss�vW@�\/2        )��P	�5If&��AN*


batch_loss[�V@�+^Z        )��P	z7yf&��AO*


batch_loss�bW@��I        )��P	� �f&��AP*


batch_loss�W@�A��        )��P	م�f&��AQ*


batch_los