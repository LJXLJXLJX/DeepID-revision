       �K"	  �����Abrain.Event:2F����     ����	�İ����A"ǳ
z
input/xPlaceholder*
dtype0*$
shape:���������77*/
_output_shapes
:���������77
l
input/yPlaceholder*
dtype0*
shape:����������
*(
_output_shapes
:����������

�
+Conv_layer_1/weights/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
o
*Conv_layer_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_1/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_1/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
)Conv_layer_1/weights/truncated_normal/mulMul5Conv_layer_1/weights/truncated_normal/TruncatedNormal,Conv_layer_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
:
�
%Conv_layer_1/weights/truncated_normalAdd)Conv_layer_1/weights/truncated_normal/mul*Conv_layer_1/weights/truncated_normal/mean*
T0*&
_output_shapes
:
�
Conv_layer_1/weights/Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name *&
_output_shapes
:
�
$Conv_layer_1/weights/Variable/AssignAssignConv_layer_1/weights/Variable%Conv_layer_1/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
"Conv_layer_1/weights/Variable/readIdentityConv_layer_1/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
f
Conv_layer_1/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes
:
�
Conv_layer_1/biases/Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
#Conv_layer_1/biases/Variable/AssignAssignConv_layer_1/biases/VariableConv_layer_1/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
!Conv_layer_1/biases/Variable/readIdentityConv_layer_1/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
Conv_layer_1/conv2dConv2Dinput/x"Conv_layer_1/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������44
�
Conv_layer_1/addAddConv_layer_1/conv2d!Conv_layer_1/biases/Variable/read*
T0*/
_output_shapes
:���������44
e
Conv_layer_1/reluReluConv_layer_1/add*
T0*/
_output_shapes
:���������44
�
Conv_layer_1/max-poolingMaxPoolConv_layer_1/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������
�
+Conv_layer_2/weights/truncated_normal/shapeConst*%
valueB"         (   *
dtype0*
_output_shapes
:
o
*Conv_layer_2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_2/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_2/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_2/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:(
�
)Conv_layer_2/weights/truncated_normal/mulMul5Conv_layer_2/weights/truncated_normal/TruncatedNormal,Conv_layer_2/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(
�
%Conv_layer_2/weights/truncated_normalAdd)Conv_layer_2/weights/truncated_normal/mul*Conv_layer_2/weights/truncated_normal/mean*
T0*&
_output_shapes
:(
�
Conv_layer_2/weights/Variable
VariableV2*
shape:(*
dtype0*
	container *
shared_name *&
_output_shapes
:(
�
$Conv_layer_2/weights/Variable/AssignAssignConv_layer_2/weights/Variable%Conv_layer_2/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
"Conv_layer_2/weights/Variable/readIdentityConv_layer_2/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
f
Conv_layer_2/biases/zerosConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Conv_layer_2/biases/Variable
VariableV2*
shape:(*
dtype0*
	container *
shared_name *
_output_shapes
:(
�
#Conv_layer_2/biases/Variable/AssignAssignConv_layer_2/biases/VariableConv_layer_2/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
!Conv_layer_2/biases/Variable/readIdentityConv_layer_2/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
Conv_layer_2/conv2dConv2DConv_layer_1/max-pooling"Conv_layer_2/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������(
�
Conv_layer_2/addAddConv_layer_2/conv2d!Conv_layer_2/biases/Variable/read*
T0*/
_output_shapes
:���������(
e
Conv_layer_2/reluReluConv_layer_2/add*
T0*/
_output_shapes
:���������(
�
Conv_layer_2/max-poolingMaxPoolConv_layer_2/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������(
�
+Conv_layer_3/weights/truncated_normal/shapeConst*%
valueB"      (   <   *
dtype0*
_output_shapes
:
o
*Conv_layer_3/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_3/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_3/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_3/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:(<
�
)Conv_layer_3/weights/truncated_normal/mulMul5Conv_layer_3/weights/truncated_normal/TruncatedNormal,Conv_layer_3/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(<
�
%Conv_layer_3/weights/truncated_normalAdd)Conv_layer_3/weights/truncated_normal/mul*Conv_layer_3/weights/truncated_normal/mean*
T0*&
_output_shapes
:(<
�
Conv_layer_3/weights/Variable
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *&
_output_shapes
:(<
�
$Conv_layer_3/weights/Variable/AssignAssignConv_layer_3/weights/Variable%Conv_layer_3/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
"Conv_layer_3/weights/Variable/readIdentityConv_layer_3/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
f
Conv_layer_3/biases/zerosConst*
valueB<*    *
dtype0*
_output_shapes
:<
�
Conv_layer_3/biases/Variable
VariableV2*
shape:<*
dtype0*
	container *
shared_name *
_output_shapes
:<
�
#Conv_layer_3/biases/Variable/AssignAssignConv_layer_3/biases/VariableConv_layer_3/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
!Conv_layer_3/biases/Variable/readIdentityConv_layer_3/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
Conv_layer_3/conv2dConv2DConv_layer_2/max-pooling"Conv_layer_3/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������

<
�
Conv_layer_3/addAddConv_layer_3/conv2d!Conv_layer_3/biases/Variable/read*
T0*/
_output_shapes
:���������

<
e
Conv_layer_3/reluReluConv_layer_3/add*
T0*/
_output_shapes
:���������

<
�
Conv_layer_3/max-poolingMaxPoolConv_layer_3/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������<
�
+Conv_layer_4/weights/truncated_normal/shapeConst*%
valueB"      <   P   *
dtype0*
_output_shapes
:
o
*Conv_layer_4/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_4/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_4/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_4/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:<P
�
)Conv_layer_4/weights/truncated_normal/mulMul5Conv_layer_4/weights/truncated_normal/TruncatedNormal,Conv_layer_4/weights/truncated_normal/stddev*
T0*&
_output_shapes
:<P
�
%Conv_layer_4/weights/truncated_normalAdd)Conv_layer_4/weights/truncated_normal/mul*Conv_layer_4/weights/truncated_normal/mean*
T0*&
_output_shapes
:<P
�
Conv_layer_4/weights/Variable
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *&
_output_shapes
:<P
�
$Conv_layer_4/weights/Variable/AssignAssignConv_layer_4/weights/Variable%Conv_layer_4/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
"Conv_layer_4/weights/Variable/readIdentityConv_layer_4/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
f
Conv_layer_4/biases/zerosConst*
valueBP*    *
dtype0*
_output_shapes
:P
�
Conv_layer_4/biases/Variable
VariableV2*
shape:P*
dtype0*
	container *
shared_name *
_output_shapes
:P
�
#Conv_layer_4/biases/Variable/AssignAssignConv_layer_4/biases/VariableConv_layer_4/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
!Conv_layer_4/biases/Variable/readIdentityConv_layer_4/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
Conv_layer_4/conv2dConv2DConv_layer_3/max-pooling"Conv_layer_4/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������P
�
Conv_layer_4/addAddConv_layer_4/conv2d!Conv_layer_4/biases/Variable/read*
T0*/
_output_shapes
:���������P
e
Conv_layer_4/reluReluConv_layer_4/add*
T0*/
_output_shapes
:���������P
f
DeepID1/Reshape/shapeConst*
valueB"�����  *
dtype0*
_output_shapes
:
�
DeepID1/ReshapeReshapeConv_layer_3/max-poolingDeepID1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
h
DeepID1/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
DeepID1/Reshape_1ReshapeConv_layer_4/reluDeepID1/Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������

w
&DeepID1/weights/truncated_normal/shapeConst*
valueB"�  �   *
dtype0*
_output_shapes
:
j
%DeepID1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
'DeepID1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
0DeepID1/weights/truncated_normal/TruncatedNormalTruncatedNormal&DeepID1/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
��
�
$DeepID1/weights/truncated_normal/mulMul0DeepID1/weights/truncated_normal/TruncatedNormal'DeepID1/weights/truncated_normal/stddev*
T0* 
_output_shapes
:
��
�
 DeepID1/weights/truncated_normalAdd$DeepID1/weights/truncated_normal/mul%DeepID1/weights/truncated_normal/mean*
T0* 
_output_shapes
:
��
�
DeepID1/weights/Variable
VariableV2*
shape:
��*
dtype0*
	container *
shared_name * 
_output_shapes
:
��
�
DeepID1/weights/Variable/AssignAssignDeepID1/weights/Variable DeepID1/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/readIdentityDeepID1/weights/Variable*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
y
(DeepID1/weights_1/truncated_normal/shapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
l
'DeepID1/weights_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
)DeepID1/weights_1/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
2DeepID1/weights_1/truncated_normal/TruncatedNormalTruncatedNormal(DeepID1/weights_1/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/truncated_normal/mulMul2DeepID1/weights_1/truncated_normal/TruncatedNormal)DeepID1/weights_1/truncated_normal/stddev*
T0* 
_output_shapes
:
�
�
�
"DeepID1/weights_1/truncated_normalAdd&DeepID1/weights_1/truncated_normal/mul'DeepID1/weights_1/truncated_normal/mean*
T0* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name * 
_output_shapes
:
�
�
�
!DeepID1/weights_1/Variable/AssignAssignDeepID1/weights_1/Variable"DeepID1/weights_1/truncated_normal*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable/readIdentityDeepID1/weights_1/Variable*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
c
DeepID1/biases/zerosConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DeepID1/biases/Variable
VariableV2*
shape:�*
dtype0*
	container *
shared_name *
_output_shapes	
:�
�
DeepID1/biases/Variable/AssignAssignDeepID1/biases/VariableDeepID1/biases/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/readIdentityDeepID1/biases/Variable*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/MatMulMatMulDeepID1/ReshapeDeepID1/weights/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
�
DeepID1/MatMul_1MatMulDeepID1/Reshape_1DeepID1/weights_1/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
g
DeepID1/addAddDeepID1/MatMulDeepID1/MatMul_1*
T0*(
_output_shapes
:����������
r
DeepID1/add_1AddDeepID1/addDeepID1/biases/Variable/read*
T0*(
_output_shapes
:����������
V
DeepID1/ReluReluDeepID1/add_1*
T0*(
_output_shapes
:����������
}
,loss/nn_layer/weights/truncated_normal/shapeConst*
valueB"�     *
dtype0*
_output_shapes
:
p
+loss/nn_layer/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
-loss/nn_layer/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
6loss/nn_layer/weights/truncated_normal/TruncatedNormalTruncatedNormal,loss/nn_layer/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
��

�
*loss/nn_layer/weights/truncated_normal/mulMul6loss/nn_layer/weights/truncated_normal/TruncatedNormal-loss/nn_layer/weights/truncated_normal/stddev*
T0* 
_output_shapes
:
��

�
&loss/nn_layer/weights/truncated_normalAdd*loss/nn_layer/weights/truncated_normal/mul+loss/nn_layer/weights/truncated_normal/mean*
T0* 
_output_shapes
:
��

�
loss/nn_layer/weights/Variable
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name * 
_output_shapes
:
��

�
%loss/nn_layer/weights/Variable/AssignAssignloss/nn_layer/weights/Variable&loss/nn_layer/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
#loss/nn_layer/weights/Variable/readIdentityloss/nn_layer/weights/Variable*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

i
loss/nn_layer/biases/zerosConst*
valueB�
*    *
dtype0*
_output_shapes	
:�

�
loss/nn_layer/biases/Variable
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *
_output_shapes	
:�

�
$loss/nn_layer/biases/Variable/AssignAssignloss/nn_layer/biases/Variableloss/nn_layer/biases/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
"loss/nn_layer/biases/Variable/readIdentityloss/nn_layer/biases/Variable*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
loss/nn_layer/Wx_plus_b/MatMulMatMulDeepID1/Relu#loss/nn_layer/weights/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������

�
loss/nn_layer/Wx_plus_b/addAddloss/nn_layer/Wx_plus_b/MatMul"loss/nn_layer/biases/Variable/read*
T0*(
_output_shapes
:����������

K
	loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e

loss/ShapeShapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
M
loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
loss/Shape_1Shapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
L

loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
I
loss/SubSubloss/Rank_1
loss/Sub/y*
T0*
_output_shapes
: 
\
loss/Slice/beginPackloss/Sub*
N*
T0*

axis *
_output_shapes
:
Y
loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
v

loss/SliceSliceloss/Shape_1loss/Slice/beginloss/Slice/size*
T0*
Index0*
_output_shapes
:
g
loss/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
R
loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
loss/concatConcatV2loss/concat/values_0
loss/Sliceloss/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
�
loss/ReshapeReshapeloss/nn_layer/Wx_plus_b/addloss/concat*
T0*
Tshape0*0
_output_shapes
:������������������
M
loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
S
loss/Shape_2Shapeinput/y*
T0*
out_type0*
_output_shapes
:
N
loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
M

loss/Sub_1Subloss/Rank_2loss/Sub_1/y*
T0*
_output_shapes
: 
`
loss/Slice_1/beginPack
loss/Sub_1*
N*
T0*

axis *
_output_shapes
:
[
loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
|
loss/Slice_1Sliceloss/Shape_2loss/Slice_1/beginloss/Slice_1/size*
T0*
Index0*
_output_shapes
:
i
loss/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
T
loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
loss/concat_1ConcatV2loss/concat_1/values_0loss/Slice_1loss/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
z
loss/Reshape_1Reshapeinput/yloss/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
"loss/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsloss/Reshapeloss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
N
loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
K

loss/Sub_2Sub	loss/Rankloss/Sub_2/y*
T0*
_output_shapes
: 
\
loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
_
loss/Slice_2/sizePack
loss/Sub_2*
N*
T0*

axis *
_output_shapes
:
�
loss/Slice_2Slice
loss/Shapeloss/Slice_2/beginloss/Slice_2/size*
T0*
Index0*#
_output_shapes
:���������
�
loss/Reshape_2Reshape"loss/SoftmaxCrossEntropyWithLogitsloss/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
k
	loss/MeanMeanloss/Reshape_2
loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
X
loss/loss/tagsConst*
valueB B	loss/loss*
dtype0*
_output_shapes
: 
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
T0*
_output_shapes
: 
n
,accuracy/correct_prediction/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
"accuracy/correct_prediction/ArgMaxArgMaxloss/nn_layer/Wx_plus_b/add,accuracy/correct_prediction/ArgMax/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
p
.accuracy/correct_prediction/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
$accuracy/correct_prediction/ArgMax_1ArgMaxinput/y.accuracy/correct_prediction/ArgMax_1/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
�
!accuracy/correct_prediction/EqualEqual"accuracy/correct_prediction/ArgMax$accuracy/correct_prediction/ArgMax_1*
T0	*#
_output_shapes
:���������
~
accuracy/accuracy/CastCast!accuracy/correct_prediction/Equal*

SrcT0
*

DstT0*#
_output_shapes
:���������
a
accuracy/accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
accuracy/accuracy/MeanMeanaccuracy/accuracy/Castaccuracy/accuracy/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
l
accuracy/accuracy_1/tagsConst*$
valueB Baccuracy/accuracy_1*
dtype0*
_output_shapes
: 
w
accuracy/accuracy_1ScalarSummaryaccuracy/accuracy_1/tagsaccuracy/accuracy/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
r
$train/gradients/loss/Mean_grad/ShapeShapeloss/Reshape_2*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
t
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Reshape_2*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
)train/gradients/loss/Reshape_2_grad/ShapeShape"loss/SoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
+train/gradients/loss/Reshape_2_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv)train/gradients/loss/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
train/gradients/zeros_like	ZerosLike$loss/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
Ftrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Btrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims+train/gradients/loss/Reshape_2_grad/ReshapeFtrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:���������
�
;train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mulMulBtrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims$loss/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
'train/gradients/loss/Reshape_grad/ShapeShapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
�
)train/gradients/loss/Reshape_grad/ReshapeReshape;train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul'train/gradients/loss/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������

�
6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ShapeShapeloss/nn_layer/Wx_plus_b/MatMul*
T0*
out_type0*
_output_shapes
:
�
8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1Const*
valueB:�
*
dtype0*
_output_shapes
:
�
Ftrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4train/gradients/loss/nn_layer/Wx_plus_b/add_grad/SumSum)train/gradients/loss/Reshape_grad/ReshapeFtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ReshapeReshape4train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������

�
6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum_1Sum)train/gradients/loss/Reshape_grad/ReshapeHtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
:train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1Reshape6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum_18train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�

�
Atrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_depsNoOp9^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape;^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1
�
Itrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependencyIdentity8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ReshapeB^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape*(
_output_shapes
:����������

�
Ktrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency_1Identity:train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1B^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1*
_output_shapes	
:�

�
:train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMulMatMulItrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency#loss/nn_layer/weights/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
<train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1MatMulDeepID1/ReluItrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��

�
Dtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp;^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul=^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1
�
Ltrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity:train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMulE^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ntrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity<train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1E^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1* 
_output_shapes
:
��

�
*train/gradients/DeepID1/Relu_grad/ReluGradReluGradLtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependencyDeepID1/Relu*
T0*(
_output_shapes
:����������
s
(train/gradients/DeepID1/add_1_grad/ShapeShapeDeepID1/add*
T0*
out_type0*
_output_shapes
:
u
*train/gradients/DeepID1/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
8train/gradients/DeepID1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs(train/gradients/DeepID1/add_1_grad/Shape*train/gradients/DeepID1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&train/gradients/DeepID1/add_1_grad/SumSum*train/gradients/DeepID1/Relu_grad/ReluGrad8train/gradients/DeepID1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
*train/gradients/DeepID1/add_1_grad/ReshapeReshape&train/gradients/DeepID1/add_1_grad/Sum(train/gradients/DeepID1/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
(train/gradients/DeepID1/add_1_grad/Sum_1Sum*train/gradients/DeepID1/Relu_grad/ReluGrad:train/gradients/DeepID1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
,train/gradients/DeepID1/add_1_grad/Reshape_1Reshape(train/gradients/DeepID1/add_1_grad/Sum_1*train/gradients/DeepID1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
3train/gradients/DeepID1/add_1_grad/tuple/group_depsNoOp+^train/gradients/DeepID1/add_1_grad/Reshape-^train/gradients/DeepID1/add_1_grad/Reshape_1
�
;train/gradients/DeepID1/add_1_grad/tuple/control_dependencyIdentity*train/gradients/DeepID1/add_1_grad/Reshape4^train/gradients/DeepID1/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/add_1_grad/Reshape*(
_output_shapes
:����������
�
=train/gradients/DeepID1/add_1_grad/tuple/control_dependency_1Identity,train/gradients/DeepID1/add_1_grad/Reshape_14^train/gradients/DeepID1/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/add_1_grad/Reshape_1*
_output_shapes	
:�
t
&train/gradients/DeepID1/add_grad/ShapeShapeDeepID1/MatMul*
T0*
out_type0*
_output_shapes
:
x
(train/gradients/DeepID1/add_grad/Shape_1ShapeDeepID1/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
6train/gradients/DeepID1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/DeepID1/add_grad/Shape(train/gradients/DeepID1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/DeepID1/add_grad/SumSum;train/gradients/DeepID1/add_1_grad/tuple/control_dependency6train/gradients/DeepID1/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
(train/gradients/DeepID1/add_grad/ReshapeReshape$train/gradients/DeepID1/add_grad/Sum&train/gradients/DeepID1/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/DeepID1/add_grad/Sum_1Sum;train/gradients/DeepID1/add_1_grad/tuple/control_dependency8train/gradients/DeepID1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
*train/gradients/DeepID1/add_grad/Reshape_1Reshape&train/gradients/DeepID1/add_grad/Sum_1(train/gradients/DeepID1/add_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
1train/gradients/DeepID1/add_grad/tuple/group_depsNoOp)^train/gradients/DeepID1/add_grad/Reshape+^train/gradients/DeepID1/add_grad/Reshape_1
�
9train/gradients/DeepID1/add_grad/tuple/control_dependencyIdentity(train/gradients/DeepID1/add_grad/Reshape2^train/gradients/DeepID1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/DeepID1/add_grad/Reshape*(
_output_shapes
:����������
�
;train/gradients/DeepID1/add_grad/tuple/control_dependency_1Identity*train/gradients/DeepID1/add_grad/Reshape_12^train/gradients/DeepID1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/add_grad/Reshape_1*(
_output_shapes
:����������
�
*train/gradients/DeepID1/MatMul_grad/MatMulMatMul9train/gradients/DeepID1/add_grad/tuple/control_dependencyDeepID1/weights/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
,train/gradients/DeepID1/MatMul_grad/MatMul_1MatMulDeepID1/Reshape9train/gradients/DeepID1/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��
�
4train/gradients/DeepID1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/DeepID1/MatMul_grad/MatMul-^train/gradients/DeepID1/MatMul_grad/MatMul_1
�
<train/gradients/DeepID1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/DeepID1/MatMul_grad/MatMul5^train/gradients/DeepID1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/DeepID1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/DeepID1/MatMul_grad/MatMul_15^train/gradients/DeepID1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
,train/gradients/DeepID1/MatMul_1_grad/MatMulMatMul;train/gradients/DeepID1/add_grad/tuple/control_dependency_1DeepID1/weights_1/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������

�
.train/gradients/DeepID1/MatMul_1_grad/MatMul_1MatMulDeepID1/Reshape_1;train/gradients/DeepID1/add_grad/tuple/control_dependency_1*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
�
�
�
6train/gradients/DeepID1/MatMul_1_grad/tuple/group_depsNoOp-^train/gradients/DeepID1/MatMul_1_grad/MatMul/^train/gradients/DeepID1/MatMul_1_grad/MatMul_1
�
>train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependencyIdentity,train/gradients/DeepID1/MatMul_1_grad/MatMul7^train/gradients/DeepID1/MatMul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/MatMul_1_grad/MatMul*(
_output_shapes
:����������

�
@train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency_1Identity.train/gradients/DeepID1/MatMul_1_grad/MatMul_17^train/gradients/DeepID1/MatMul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/DeepID1/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�
�
�
*train/gradients/DeepID1/Reshape_grad/ShapeShapeConv_layer_3/max-pooling*
T0*
out_type0*
_output_shapes
:
�
,train/gradients/DeepID1/Reshape_grad/ReshapeReshape<train/gradients/DeepID1/MatMul_grad/tuple/control_dependency*train/gradients/DeepID1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������<
}
,train/gradients/DeepID1/Reshape_1_grad/ShapeShapeConv_layer_4/relu*
T0*
out_type0*
_output_shapes
:
�
.train/gradients/DeepID1/Reshape_1_grad/ReshapeReshape>train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency,train/gradients/DeepID1/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������P
�
/train/gradients/Conv_layer_4/relu_grad/ReluGradReluGrad.train/gradients/DeepID1/Reshape_1_grad/ReshapeConv_layer_4/relu*
T0*/
_output_shapes
:���������P
~
+train/gradients/Conv_layer_4/add_grad/ShapeShapeConv_layer_4/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_4/add_grad/Shape_1Const*
valueB:P*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_4/add_grad/Shape-train/gradients/Conv_layer_4/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_4/add_grad/SumSum/train/gradients/Conv_layer_4/relu_grad/ReluGrad;train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_4/add_grad/ReshapeReshape)train/gradients/Conv_layer_4/add_grad/Sum+train/gradients/Conv_layer_4/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������P
�
+train/gradients/Conv_layer_4/add_grad/Sum_1Sum/train/gradients/Conv_layer_4/relu_grad/ReluGrad=train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_4/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_4/add_grad/Sum_1-train/gradients/Conv_layer_4/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:P
�
6train/gradients/Conv_layer_4/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_4/add_grad/Reshape0^train/gradients/Conv_layer_4/add_grad/Reshape_1
�
>train/gradients/Conv_layer_4/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_4/add_grad/Reshape7^train/gradients/Conv_layer_4/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_4/add_grad/Reshape*/
_output_shapes
:���������P
�
@train/gradients/Conv_layer_4/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_4/add_grad/Reshape_17^train/gradients/Conv_layer_4/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_4/add_grad/Reshape_1*
_output_shapes
:P
�
/train/gradients/Conv_layer_4/conv2d_grad/ShapeNShapeNConv_layer_3/max-pooling"Conv_layer_4/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_4/conv2d_grad/ShapeN"Conv_layer_4/weights/Variable/read>train/gradients/Conv_layer_4/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_3/max-pooling1train/gradients/Conv_layer_4/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_4/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_4/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_4/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������<
�
Ctrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_4/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:<P
�
train/gradients/AddNAddN,train/gradients/DeepID1/Reshape_grad/ReshapeAtrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency*
N*
T0*?
_class5
31loc:@train/gradients/DeepID1/Reshape_grad/Reshape*/
_output_shapes
:���������<
�
9train/gradients/Conv_layer_3/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_3/reluConv_layer_3/max-poolingtrain/gradients/AddN*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������

<
�
/train/gradients/Conv_layer_3/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_3/max-pooling_grad/MaxPoolGradConv_layer_3/relu*
T0*/
_output_shapes
:���������

<
~
+train/gradients/Conv_layer_3/add_grad/ShapeShapeConv_layer_3/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_3/add_grad/Shape_1Const*
valueB:<*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_3/add_grad/Shape-train/gradients/Conv_layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_3/add_grad/SumSum/train/gradients/Conv_layer_3/relu_grad/ReluGrad;train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_3/add_grad/ReshapeReshape)train/gradients/Conv_layer_3/add_grad/Sum+train/gradients/Conv_layer_3/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������

<
�
+train/gradients/Conv_layer_3/add_grad/Sum_1Sum/train/gradients/Conv_layer_3/relu_grad/ReluGrad=train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_3/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_3/add_grad/Sum_1-train/gradients/Conv_layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:<
�
6train/gradients/Conv_layer_3/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_3/add_grad/Reshape0^train/gradients/Conv_layer_3/add_grad/Reshape_1
�
>train/gradients/Conv_layer_3/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_3/add_grad/Reshape7^train/gradients/Conv_layer_3/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_3/add_grad/Reshape*/
_output_shapes
:���������

<
�
@train/gradients/Conv_layer_3/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_3/add_grad/Reshape_17^train/gradients/Conv_layer_3/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_3/add_grad/Reshape_1*
_output_shapes
:<
�
/train/gradients/Conv_layer_3/conv2d_grad/ShapeNShapeNConv_layer_2/max-pooling"Conv_layer_3/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_3/conv2d_grad/ShapeN"Conv_layer_3/weights/Variable/read>train/gradients/Conv_layer_3/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_2/max-pooling1train/gradients/Conv_layer_3/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_3/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_3/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_3/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������(
�
Ctrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_3/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:(<
�
9train/gradients/Conv_layer_2/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_2/reluConv_layer_2/max-poolingAtrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������(
�
/train/gradients/Conv_layer_2/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_2/max-pooling_grad/MaxPoolGradConv_layer_2/relu*
T0*/
_output_shapes
:���������(
~
+train/gradients/Conv_layer_2/add_grad/ShapeShapeConv_layer_2/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_2/add_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_2/add_grad/Shape-train/gradients/Conv_layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_2/add_grad/SumSum/train/gradients/Conv_layer_2/relu_grad/ReluGrad;train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_2/add_grad/ReshapeReshape)train/gradients/Conv_layer_2/add_grad/Sum+train/gradients/Conv_layer_2/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������(
�
+train/gradients/Conv_layer_2/add_grad/Sum_1Sum/train/gradients/Conv_layer_2/relu_grad/ReluGrad=train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_2/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_2/add_grad/Sum_1-train/gradients/Conv_layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:(
�
6train/gradients/Conv_layer_2/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_2/add_grad/Reshape0^train/gradients/Conv_layer_2/add_grad/Reshape_1
�
>train/gradients/Conv_layer_2/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_2/add_grad/Reshape7^train/gradients/Conv_layer_2/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_2/add_grad/Reshape*/
_output_shapes
:���������(
�
@train/gradients/Conv_layer_2/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_2/add_grad/Reshape_17^train/gradients/Conv_layer_2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_2/add_grad/Reshape_1*
_output_shapes
:(
�
/train/gradients/Conv_layer_2/conv2d_grad/ShapeNShapeNConv_layer_1/max-pooling"Conv_layer_2/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_2/conv2d_grad/ShapeN"Conv_layer_2/weights/Variable/read>train/gradients/Conv_layer_2/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_1/max-pooling1train/gradients/Conv_layer_2/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_2/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_2/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_2/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Ctrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_2/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:(
�
9train/gradients/Conv_layer_1/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_1/reluConv_layer_1/max-poolingAtrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������44
�
/train/gradients/Conv_layer_1/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_1/max-pooling_grad/MaxPoolGradConv_layer_1/relu*
T0*/
_output_shapes
:���������44
~
+train/gradients/Conv_layer_1/add_grad/ShapeShapeConv_layer_1/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_1/add_grad/Shape-train/gradients/Conv_layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_1/add_grad/SumSum/train/gradients/Conv_layer_1/relu_grad/ReluGrad;train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_1/add_grad/ReshapeReshape)train/gradients/Conv_layer_1/add_grad/Sum+train/gradients/Conv_layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������44
�
+train/gradients/Conv_layer_1/add_grad/Sum_1Sum/train/gradients/Conv_layer_1/relu_grad/ReluGrad=train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_1/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_1/add_grad/Sum_1-train/gradients/Conv_layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
6train/gradients/Conv_layer_1/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_1/add_grad/Reshape0^train/gradients/Conv_layer_1/add_grad/Reshape_1
�
>train/gradients/Conv_layer_1/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_1/add_grad/Reshape7^train/gradients/Conv_layer_1/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_1/add_grad/Reshape*/
_output_shapes
:���������44
�
@train/gradients/Conv_layer_1/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_1/add_grad/Reshape_17^train/gradients/Conv_layer_1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_1/add_grad/Reshape_1*
_output_shapes
:
�
/train/gradients/Conv_layer_1/conv2d_grad/ShapeNShapeNinput/x"Conv_layer_1/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_1/conv2d_grad/ShapeN"Conv_layer_1/weights/Variable/read>train/gradients/Conv_layer_1/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/x1train/gradients/Conv_layer_1/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_1/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_1/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_1/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������77
�
Ctrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_1/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
train/beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power/readIdentitytrain/beta1_power*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/readIdentitytrain/beta2_power*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
4Conv_layer_1/weights/Variable/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
"Conv_layer_1/weights/Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
)Conv_layer_1/weights/Variable/Adam/AssignAssign"Conv_layer_1/weights/Variable/Adam4Conv_layer_1/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
'Conv_layer_1/weights/Variable/Adam/readIdentity"Conv_layer_1/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
6Conv_layer_1/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
$Conv_layer_1/weights/Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
+Conv_layer_1/weights/Variable/Adam_1/AssignAssign$Conv_layer_1/weights/Variable/Adam_16Conv_layer_1/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
)Conv_layer_1/weights/Variable/Adam_1/readIdentity$Conv_layer_1/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
3Conv_layer_1/biases/Variable/Adam/Initializer/zerosConst*
valueB*    *
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
!Conv_layer_1/biases/Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
(Conv_layer_1/biases/Variable/Adam/AssignAssign!Conv_layer_1/biases/Variable/Adam3Conv_layer_1/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
&Conv_layer_1/biases/Variable/Adam/readIdentity!Conv_layer_1/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
5Conv_layer_1/biases/Variable/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
#Conv_layer_1/biases/Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
*Conv_layer_1/biases/Variable/Adam_1/AssignAssign#Conv_layer_1/biases/Variable/Adam_15Conv_layer_1/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
(Conv_layer_1/biases/Variable/Adam_1/readIdentity#Conv_layer_1/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
4Conv_layer_2/weights/Variable/Adam/Initializer/zerosConst*%
valueB(*    *
dtype0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
"Conv_layer_2/weights/Variable/Adam
VariableV2*
shape:(*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
)Conv_layer_2/weights/Variable/Adam/AssignAssign"Conv_layer_2/weights/Variable/Adam4Conv_layer_2/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
'Conv_layer_2/weights/Variable/Adam/readIdentity"Conv_layer_2/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
6Conv_layer_2/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB(*    *
dtype0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
$Conv_layer_2/weights/Variable/Adam_1
VariableV2*
shape:(*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
+Conv_layer_2/weights/Variable/Adam_1/AssignAssign$Conv_layer_2/weights/Variable/Adam_16Conv_layer_2/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
)Conv_layer_2/weights/Variable/Adam_1/readIdentity$Conv_layer_2/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
3Conv_layer_2/biases/Variable/Adam/Initializer/zerosConst*
valueB(*    *
dtype0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
!Conv_layer_2/biases/Variable/Adam
VariableV2*
shape:(*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
(Conv_layer_2/biases/Variable/Adam/AssignAssign!Conv_layer_2/biases/Variable/Adam3Conv_layer_2/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
&Conv_layer_2/biases/Variable/Adam/readIdentity!Conv_layer_2/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
5Conv_layer_2/biases/Variable/Adam_1/Initializer/zerosConst*
valueB(*    *
dtype0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
#Conv_layer_2/biases/Variable/Adam_1
VariableV2*
shape:(*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
*Conv_layer_2/biases/Variable/Adam_1/AssignAssign#Conv_layer_2/biases/Variable/Adam_15Conv_layer_2/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
(Conv_layer_2/biases/Variable/Adam_1/readIdentity#Conv_layer_2/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
4Conv_layer_3/weights/Variable/Adam/Initializer/zerosConst*%
valueB(<*    *
dtype0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
"Conv_layer_3/weights/Variable/Adam
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
)Conv_layer_3/weights/Variable/Adam/AssignAssign"Conv_layer_3/weights/Variable/Adam4Conv_layer_3/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
'Conv_layer_3/weights/Variable/Adam/readIdentity"Conv_layer_3/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
6Conv_layer_3/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB(<*    *
dtype0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
$Conv_layer_3/weights/Variable/Adam_1
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
+Conv_layer_3/weights/Variable/Adam_1/AssignAssign$Conv_layer_3/weights/Variable/Adam_16Conv_layer_3/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
)Conv_layer_3/weights/Variable/Adam_1/readIdentity$Conv_layer_3/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
3Conv_layer_3/biases/Variable/Adam/Initializer/zerosConst*
valueB<*    *
dtype0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
!Conv_layer_3/biases/Variable/Adam
VariableV2*
shape:<*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
(Conv_layer_3/biases/Variable/Adam/AssignAssign!Conv_layer_3/biases/Variable/Adam3Conv_layer_3/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
&Conv_layer_3/biases/Variable/Adam/readIdentity!Conv_layer_3/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
5Conv_layer_3/biases/Variable/Adam_1/Initializer/zerosConst*
valueB<*    *
dtype0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
#Conv_layer_3/biases/Variable/Adam_1
VariableV2*
shape:<*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
*Conv_layer_3/biases/Variable/Adam_1/AssignAssign#Conv_layer_3/biases/Variable/Adam_15Conv_layer_3/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
(Conv_layer_3/biases/Variable/Adam_1/readIdentity#Conv_layer_3/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
4Conv_layer_4/weights/Variable/Adam/Initializer/zerosConst*%
valueB<P*    *
dtype0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
"Conv_layer_4/weights/Variable/Adam
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
)Conv_layer_4/weights/Variable/Adam/AssignAssign"Conv_layer_4/weights/Variable/Adam4Conv_layer_4/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
'Conv_layer_4/weights/Variable/Adam/readIdentity"Conv_layer_4/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
6Conv_layer_4/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB<P*    *
dtype0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
$Conv_layer_4/weights/Variable/Adam_1
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
+Conv_layer_4/weights/Variable/Adam_1/AssignAssign$Conv_layer_4/weights/Variable/Adam_16Conv_layer_4/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
)Conv_layer_4/weights/Variable/Adam_1/readIdentity$Conv_layer_4/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
3Conv_layer_4/biases/Variable/Adam/Initializer/zerosConst*
valueBP*    *
dtype0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
!Conv_layer_4/biases/Variable/Adam
VariableV2*
shape:P*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
(Conv_layer_4/biases/Variable/Adam/AssignAssign!Conv_layer_4/biases/Variable/Adam3Conv_layer_4/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
&Conv_layer_4/biases/Variable/Adam/readIdentity!Conv_layer_4/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
5Conv_layer_4/biases/Variable/Adam_1/Initializer/zerosConst*
valueBP*    *
dtype0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
#Conv_layer_4/biases/Variable/Adam_1
VariableV2*
shape:P*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
*Conv_layer_4/biases/Variable/Adam_1/AssignAssign#Conv_layer_4/biases/Variable/Adam_15Conv_layer_4/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
(Conv_layer_4/biases/Variable/Adam_1/readIdentity#Conv_layer_4/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
/DeepID1/weights/Variable/Adam/Initializer/zerosConst*
valueB
��*    *
dtype0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/Adam
VariableV2*
shape:
��*
dtype0*
	container *
shared_name *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
$DeepID1/weights/Variable/Adam/AssignAssignDeepID1/weights/Variable/Adam/DeepID1/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
"DeepID1/weights/Variable/Adam/readIdentityDeepID1/weights/Variable/Adam*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
1DeepID1/weights/Variable/Adam_1/Initializer/zerosConst*
valueB
��*    *
dtype0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/Adam_1
VariableV2*
shape:
��*
dtype0*
	container *
shared_name *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
&DeepID1/weights/Variable/Adam_1/AssignAssignDeepID1/weights/Variable/Adam_11DeepID1/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
$DeepID1/weights/Variable/Adam_1/readIdentityDeepID1/weights/Variable/Adam_1*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
1DeepID1/weights_1/Variable/Adam/Initializer/zerosConst*
valueB
�
�*    *
dtype0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable/Adam
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/Variable/Adam/AssignAssignDeepID1/weights_1/Variable/Adam1DeepID1/weights_1/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
$DeepID1/weights_1/Variable/Adam/readIdentityDeepID1/weights_1/Variable/Adam*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
3DeepID1/weights_1/Variable/Adam_1/Initializer/zerosConst*
valueB
�
�*    *
dtype0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
!DeepID1/weights_1/Variable/Adam_1
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
(DeepID1/weights_1/Variable/Adam_1/AssignAssign!DeepID1/weights_1/Variable/Adam_13DeepID1/weights_1/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/Variable/Adam_1/readIdentity!DeepID1/weights_1/Variable/Adam_1*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
.DeepID1/biases/Variable/Adam/Initializer/zerosConst*
valueB�*    *
dtype0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/Adam
VariableV2*
shape:�*
dtype0*
	container *
shared_name **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
#DeepID1/biases/Variable/Adam/AssignAssignDeepID1/biases/Variable/Adam.DeepID1/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
!DeepID1/biases/Variable/Adam/readIdentityDeepID1/biases/Variable/Adam*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
0DeepID1/biases/Variable/Adam_1/Initializer/zerosConst*
valueB�*    *
dtype0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/Adam_1
VariableV2*
shape:�*
dtype0*
	container *
shared_name **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
%DeepID1/biases/Variable/Adam_1/AssignAssignDeepID1/biases/Variable/Adam_10DeepID1/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
#DeepID1/biases/Variable/Adam_1/readIdentityDeepID1/biases/Variable/Adam_1*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
5loss/nn_layer/weights/Variable/Adam/Initializer/zerosConst*
valueB
��
*    *
dtype0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
#loss/nn_layer/weights/Variable/Adam
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
*loss/nn_layer/weights/Variable/Adam/AssignAssign#loss/nn_layer/weights/Variable/Adam5loss/nn_layer/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
(loss/nn_layer/weights/Variable/Adam/readIdentity#loss/nn_layer/weights/Variable/Adam*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
7loss/nn_layer/weights/Variable/Adam_1/Initializer/zerosConst*
valueB
��
*    *
dtype0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
%loss/nn_layer/weights/Variable/Adam_1
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
,loss/nn_layer/weights/Variable/Adam_1/AssignAssign%loss/nn_layer/weights/Variable/Adam_17loss/nn_layer/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
*loss/nn_layer/weights/Variable/Adam_1/readIdentity%loss/nn_layer/weights/Variable/Adam_1*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
4loss/nn_layer/biases/Variable/Adam/Initializer/zerosConst*
valueB�
*    *
dtype0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
"loss/nn_layer/biases/Variable/Adam
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
)loss/nn_layer/biases/Variable/Adam/AssignAssign"loss/nn_layer/biases/Variable/Adam4loss/nn_layer/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
'loss/nn_layer/biases/Variable/Adam/readIdentity"loss/nn_layer/biases/Variable/Adam*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
6loss/nn_layer/biases/Variable/Adam_1/Initializer/zerosConst*
valueB�
*    *
dtype0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
$loss/nn_layer/biases/Variable/Adam_1
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
+loss/nn_layer/biases/Variable/Adam_1/AssignAssign$loss/nn_layer/biases/Variable/Adam_16loss/nn_layer/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
)loss/nn_layer/biases/Variable/Adam_1/readIdentity$loss/nn_layer/biases/Variable/Adam_1*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

]
train/Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
9train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam	ApplyAdamConv_layer_1/weights/Variable"Conv_layer_1/weights/Variable/Adam$Conv_layer_1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
8train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam	ApplyAdamConv_layer_1/biases/Variable!Conv_layer_1/biases/Variable/Adam#Conv_layer_1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_1/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
9train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam	ApplyAdamConv_layer_2/weights/Variable"Conv_layer_2/weights/Variable/Adam$Conv_layer_2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
8train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam	ApplyAdamConv_layer_2/biases/Variable!Conv_layer_2/biases/Variable/Adam#Conv_layer_2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_2/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
9train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam	ApplyAdamConv_layer_3/weights/Variable"Conv_layer_3/weights/Variable/Adam$Conv_layer_3/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
8train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam	ApplyAdamConv_layer_3/biases/Variable!Conv_layer_3/biases/Variable/Adam#Conv_layer_3/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_3/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
9train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam	ApplyAdamConv_layer_4/weights/Variable"Conv_layer_4/weights/Variable/Adam$Conv_layer_4/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
8train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam	ApplyAdamConv_layer_4/biases/Variable!Conv_layer_4/biases/Variable/Adam#Conv_layer_4/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_4/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
4train/Adam/update_DeepID1/weights/Variable/ApplyAdam	ApplyAdamDeepID1/weights/VariableDeepID1/weights/Variable/AdamDeepID1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/DeepID1/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
6train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam	ApplyAdamDeepID1/weights_1/VariableDeepID1/weights_1/Variable/Adam!DeepID1/weights_1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
3train/Adam/update_DeepID1/biases/Variable/ApplyAdam	ApplyAdamDeepID1/biases/VariableDeepID1/biases/Variable/AdamDeepID1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/DeepID1/add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
:train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam	ApplyAdamloss/nn_layer/weights/Variable#loss/nn_layer/weights/Variable/Adam%loss/nn_layer/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
9train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam	ApplyAdamloss/nn_layer/biases/Variable"loss/nn_layer/biases/Variable/Adam$loss/nn_layer/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonKtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*
validate_shape(*
use_locking( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*
validate_shape(*
use_locking( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�

train/AdamNoOp:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
c
Merge/MergeSummaryMergeSummary	loss/lossaccuracy/accuracy_1*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�

value�
B�
)BConv_layer_1/biases/VariableB!Conv_layer_1/biases/Variable/AdamB#Conv_layer_1/biases/Variable/Adam_1BConv_layer_1/weights/VariableB"Conv_layer_1/weights/Variable/AdamB$Conv_layer_1/weights/Variable/Adam_1BConv_layer_2/biases/VariableB!Conv_layer_2/biases/Variable/AdamB#Conv_layer_2/biases/Variable/Adam_1BConv_layer_2/weights/VariableB"Conv_layer_2/weights/Variable/AdamB$Conv_layer_2/weights/Variable/Adam_1BConv_layer_3/biases/VariableB!Conv_layer_3/biases/Variable/AdamB#Conv_layer_3/biases/Variable/Adam_1BConv_layer_3/weights/VariableB"Conv_layer_3/weights/Variable/AdamB$Conv_layer_3/weights/Variable/Adam_1BConv_layer_4/biases/VariableB!Conv_layer_4/biases/Variable/AdamB#Conv_layer_4/biases/Variable/Adam_1BConv_layer_4/weights/VariableB"Conv_layer_4/weights/Variable/AdamB$Conv_layer_4/weights/Variable/Adam_1BDeepID1/biases/VariableBDeepID1/biases/Variable/AdamBDeepID1/biases/Variable/Adam_1BDeepID1/weights/VariableBDeepID1/weights/Variable/AdamBDeepID1/weights/Variable/Adam_1BDeepID1/weights_1/VariableBDeepID1/weights_1/Variable/AdamB!DeepID1/weights_1/Variable/Adam_1Bloss/nn_layer/biases/VariableB"loss/nn_layer/biases/Variable/AdamB$loss/nn_layer/biases/Variable/Adam_1Bloss/nn_layer/weights/VariableB#loss/nn_layer/weights/Variable/AdamB%loss/nn_layer/weights/Variable/Adam_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:)
�
save/SaveV2/shape_and_slicesConst*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:)
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConv_layer_1/biases/Variable!Conv_layer_1/biases/Variable/Adam#Conv_layer_1/biases/Variable/Adam_1Conv_layer_1/weights/Variable"Conv_layer_1/weights/Variable/Adam$Conv_layer_1/weights/Variable/Adam_1Conv_layer_2/biases/Variable!Conv_layer_2/biases/Variable/Adam#Conv_layer_2/biases/Variable/Adam_1Conv_layer_2/weights/Variable"Conv_layer_2/weights/Variable/Adam$Conv_layer_2/weights/Variable/Adam_1Conv_layer_3/biases/Variable!Conv_layer_3/biases/Variable/Adam#Conv_layer_3/biases/Variable/Adam_1Conv_layer_3/weights/Variable"Conv_layer_3/weights/Variable/Adam$Conv_layer_3/weights/Variable/Adam_1Conv_layer_4/biases/Variable!Conv_layer_4/biases/Variable/Adam#Conv_layer_4/biases/Variable/Adam_1Conv_layer_4/weights/Variable"Conv_layer_4/weights/Variable/Adam$Conv_layer_4/weights/Variable/Adam_1DeepID1/biases/VariableDeepID1/biases/Variable/AdamDeepID1/biases/Variable/Adam_1DeepID1/weights/VariableDeepID1/weights/Variable/AdamDeepID1/weights/Variable/Adam_1DeepID1/weights_1/VariableDeepID1/weights_1/Variable/Adam!DeepID1/weights_1/Variable/Adam_1loss/nn_layer/biases/Variable"loss/nn_layer/biases/Variable/Adam$loss/nn_layer/biases/Variable/Adam_1loss/nn_layer/weights/Variable#loss/nn_layer/weights/Variable/Adam%loss/nn_layer/weights/Variable/Adam_1train/beta1_powertrain/beta2_power*7
dtypes-
+2)
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*1
value(B&BConv_layer_1/biases/Variable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignConv_layer_1/biases/Variablesave/RestoreV2*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_1/tensor_namesConst*6
value-B+B!Conv_layer_1/biases/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign!Conv_layer_1/biases/Variable/Adamsave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_2/tensor_namesConst*8
value/B-B#Conv_layer_1/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign#Conv_layer_1/biases/Variable/Adam_1save/RestoreV2_2*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_3/tensor_namesConst*2
value)B'BConv_layer_1/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3AssignConv_layer_1/weights/Variablesave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_4/tensor_namesConst*7
value.B,B"Conv_layer_1/weights/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign"Conv_layer_1/weights/Variable/Adamsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_5/tensor_namesConst*9
value0B.B$Conv_layer_1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign$Conv_layer_1/weights/Variable/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_6/tensor_namesConst*1
value(B&BConv_layer_2/biases/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6AssignConv_layer_2/biases/Variablesave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_7/tensor_namesConst*6
value-B+B!Conv_layer_2/biases/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assign!Conv_layer_2/biases/Variable/Adamsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_8/tensor_namesConst*8
value/B-B#Conv_layer_2/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assign#Conv_layer_2/biases/Variable/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_9/tensor_namesConst*2
value)B'BConv_layer_2/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9AssignConv_layer_2/weights/Variablesave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_10/tensor_namesConst*7
value.B,B"Conv_layer_2/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign"Conv_layer_2/weights/Variable/Adamsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_11/tensor_namesConst*9
value0B.B$Conv_layer_2/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assign$Conv_layer_2/weights/Variable/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_12/tensor_namesConst*1
value(B&BConv_layer_3/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12AssignConv_layer_3/biases/Variablesave/RestoreV2_12*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_13/tensor_namesConst*6
value-B+B!Conv_layer_3/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assign!Conv_layer_3/biases/Variable/Adamsave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_14/tensor_namesConst*8
value/B-B#Conv_layer_3/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign#Conv_layer_3/biases/Variable/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_15/tensor_namesConst*2
value)B'BConv_layer_3/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15AssignConv_layer_3/weights/Variablesave/RestoreV2_15*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_16/tensor_namesConst*7
value.B,B"Conv_layer_3/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign"Conv_layer_3/weights/Variable/Adamsave/RestoreV2_16*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_17/tensor_namesConst*9
value0B.B$Conv_layer_3/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign$Conv_layer_3/weights/Variable/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_18/tensor_namesConst*1
value(B&BConv_layer_4/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18AssignConv_layer_4/biases/Variablesave/RestoreV2_18*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_19/tensor_namesConst*6
value-B+B!Conv_layer_4/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assign!Conv_layer_4/biases/Variable/Adamsave/RestoreV2_19*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_20/tensor_namesConst*8
value/B-B#Conv_layer_4/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign#Conv_layer_4/biases/Variable/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_21/tensor_namesConst*2
value)B'BConv_layer_4/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_21AssignConv_layer_4/weights/Variablesave/RestoreV2_21*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
save/RestoreV2_22/tensor_namesConst*7
value.B,B"Conv_layer_4/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assign"Conv_layer_4/weights/Variable/Adamsave/RestoreV2_22*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
save/RestoreV2_23/tensor_namesConst*9
value0B.B$Conv_layer_4/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_23Assign$Conv_layer_4/weights/Variable/Adam_1save/RestoreV2_23*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
~
save/RestoreV2_24/tensor_namesConst*,
value#B!BDeepID1/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_24AssignDeepID1/biases/Variablesave/RestoreV2_24*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
save/RestoreV2_25/tensor_namesConst*1
value(B&BDeepID1/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_25AssignDeepID1/biases/Variable/Adamsave/RestoreV2_25*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
save/RestoreV2_26/tensor_namesConst*3
value*B(BDeepID1/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26AssignDeepID1/biases/Variable/Adam_1save/RestoreV2_26*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�

save/RestoreV2_27/tensor_namesConst*-
value$B"BDeepID1/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27AssignDeepID1/weights/Variablesave/RestoreV2_27*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_28/tensor_namesConst*2
value)B'BDeepID1/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_28AssignDeepID1/weights/Variable/Adamsave/RestoreV2_28*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_29/tensor_namesConst*4
value+B)BDeepID1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29AssignDeepID1/weights/Variable/Adam_1save/RestoreV2_29*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_30/tensor_namesConst*/
value&B$BDeepID1/weights_1/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30AssignDeepID1/weights_1/Variablesave/RestoreV2_30*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_31/tensor_namesConst*4
value+B)BDeepID1/weights_1/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_31AssignDeepID1/weights_1/Variable/Adamsave/RestoreV2_31*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_32/tensor_namesConst*6
value-B+B!DeepID1/weights_1/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assign!DeepID1/weights_1/Variable/Adam_1save/RestoreV2_32*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_33/tensor_namesConst*2
value)B'Bloss/nn_layer/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignloss/nn_layer/biases/Variablesave/RestoreV2_33*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_34/tensor_namesConst*7
value.B,B"loss/nn_layer/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assign"loss/nn_layer/biases/Variable/Adamsave/RestoreV2_34*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_35/tensor_namesConst*9
value0B.B$loss/nn_layer/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_35Assign$loss/nn_layer/biases/Variable/Adam_1save/RestoreV2_35*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_36/tensor_namesConst*3
value*B(Bloss/nn_layer/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_36Assignloss/nn_layer/weights/Variablesave/RestoreV2_36*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
save/RestoreV2_37/tensor_namesConst*8
value/B-B#loss/nn_layer/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_37Assign#loss/nn_layer/weights/Variable/Adamsave/RestoreV2_37*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
save/RestoreV2_38/tensor_namesConst*:
value1B/B%loss/nn_layer/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_38Assign%loss/nn_layer/weights/Variable/Adam_1save/RestoreV2_38*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

x
save/RestoreV2_39/tensor_namesConst*&
valueBBtrain/beta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_39Assigntrain/beta1_powersave/RestoreV2_39*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
x
save/RestoreV2_40/tensor_namesConst*&
valueBBtrain/beta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_40Assigntrain/beta2_powersave/RestoreV2_40*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40
�
initNoOp%^Conv_layer_1/weights/Variable/Assign$^Conv_layer_1/biases/Variable/Assign%^Conv_layer_2/weights/Variable/Assign$^Conv_layer_2/biases/Variable/Assign%^Conv_layer_3/weights/Variable/Assign$^Conv_layer_3/biases/Variable/Assign%^Conv_layer_4/weights/Variable/Assign$^Conv_layer_4/biases/Variable/Assign ^DeepID1/weights/Variable/Assign"^DeepID1/weights_1/Variable/Assign^DeepID1/biases/Variable/Assign&^loss/nn_layer/weights/Variable/Assign%^loss/nn_layer/biases/Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign*^Conv_layer_1/weights/Variable/Adam/Assign,^Conv_layer_1/weights/Variable/Adam_1/Assign)^Conv_layer_1/biases/Variable/Adam/Assign+^Conv_layer_1/biases/Variable/Adam_1/Assign*^Conv_layer_2/weights/Variable/Adam/Assign,^Conv_layer_2/weights/Variable/Adam_1/Assign)^Conv_layer_2/biases/Variable/Adam/Assign+^Conv_layer_2/biases/Variable/Adam_1/Assign*^Conv_layer_3/weights/Variable/Adam/Assign,^Conv_layer_3/weights/Variable/Adam_1/Assign)^Conv_layer_3/biases/Variable/Adam/Assign+^Conv_layer_3/biases/Variable/Adam_1/Assign*^Conv_layer_4/weights/Variable/Adam/Assign,^Conv_layer_4/weights/Variable/Adam_1/Assign)^Conv_layer_4/biases/Variable/Adam/Assign+^Conv_layer_4/biases/Variable/Adam_1/Assign%^DeepID1/weights/Variable/Adam/Assign'^DeepID1/weights/Variable/Adam_1/Assign'^DeepID1/weights_1/Variable/Adam/Assign)^DeepID1/weights_1/Variable/Adam_1/Assign$^DeepID1/biases/Variable/Adam/Assign&^DeepID1/biases/Variable/Adam_1/Assign+^loss/nn_layer/weights/Variable/Adam/Assign-^loss/nn_layer/weights/Variable/Adam_1/Assign*^loss/nn_layer/biases/Variable/Adam/Assign,^loss/nn_layer/biases/Variable/Adam_1/Assign"{�(��     ��0�	Ħỻ��AJ��
�%�%
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
Ttype*1.4.12
b'unknown'ǳ
z
input/xPlaceholder*
dtype0*$
shape:���������77*/
_output_shapes
:���������77
l
input/yPlaceholder*
dtype0*
shape:����������
*(
_output_shapes
:����������

�
+Conv_layer_1/weights/truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
o
*Conv_layer_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_1/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_1/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
)Conv_layer_1/weights/truncated_normal/mulMul5Conv_layer_1/weights/truncated_normal/TruncatedNormal,Conv_layer_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
:
�
%Conv_layer_1/weights/truncated_normalAdd)Conv_layer_1/weights/truncated_normal/mul*Conv_layer_1/weights/truncated_normal/mean*
T0*&
_output_shapes
:
�
Conv_layer_1/weights/Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name *&
_output_shapes
:
�
$Conv_layer_1/weights/Variable/AssignAssignConv_layer_1/weights/Variable%Conv_layer_1/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
"Conv_layer_1/weights/Variable/readIdentityConv_layer_1/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
f
Conv_layer_1/biases/zerosConst*
valueB*    *
dtype0*
_output_shapes
:
�
Conv_layer_1/biases/Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
#Conv_layer_1/biases/Variable/AssignAssignConv_layer_1/biases/VariableConv_layer_1/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
!Conv_layer_1/biases/Variable/readIdentityConv_layer_1/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
Conv_layer_1/conv2dConv2Dinput/x"Conv_layer_1/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������44
�
Conv_layer_1/addAddConv_layer_1/conv2d!Conv_layer_1/biases/Variable/read*
T0*/
_output_shapes
:���������44
e
Conv_layer_1/reluReluConv_layer_1/add*
T0*/
_output_shapes
:���������44
�
Conv_layer_1/max-poolingMaxPoolConv_layer_1/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������
�
+Conv_layer_2/weights/truncated_normal/shapeConst*%
valueB"         (   *
dtype0*
_output_shapes
:
o
*Conv_layer_2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_2/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_2/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_2/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:(
�
)Conv_layer_2/weights/truncated_normal/mulMul5Conv_layer_2/weights/truncated_normal/TruncatedNormal,Conv_layer_2/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(
�
%Conv_layer_2/weights/truncated_normalAdd)Conv_layer_2/weights/truncated_normal/mul*Conv_layer_2/weights/truncated_normal/mean*
T0*&
_output_shapes
:(
�
Conv_layer_2/weights/Variable
VariableV2*
shape:(*
dtype0*
	container *
shared_name *&
_output_shapes
:(
�
$Conv_layer_2/weights/Variable/AssignAssignConv_layer_2/weights/Variable%Conv_layer_2/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
"Conv_layer_2/weights/Variable/readIdentityConv_layer_2/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
f
Conv_layer_2/biases/zerosConst*
valueB(*    *
dtype0*
_output_shapes
:(
�
Conv_layer_2/biases/Variable
VariableV2*
shape:(*
dtype0*
	container *
shared_name *
_output_shapes
:(
�
#Conv_layer_2/biases/Variable/AssignAssignConv_layer_2/biases/VariableConv_layer_2/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
!Conv_layer_2/biases/Variable/readIdentityConv_layer_2/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
Conv_layer_2/conv2dConv2DConv_layer_1/max-pooling"Conv_layer_2/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������(
�
Conv_layer_2/addAddConv_layer_2/conv2d!Conv_layer_2/biases/Variable/read*
T0*/
_output_shapes
:���������(
e
Conv_layer_2/reluReluConv_layer_2/add*
T0*/
_output_shapes
:���������(
�
Conv_layer_2/max-poolingMaxPoolConv_layer_2/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������(
�
+Conv_layer_3/weights/truncated_normal/shapeConst*%
valueB"      (   <   *
dtype0*
_output_shapes
:
o
*Conv_layer_3/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_3/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_3/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_3/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:(<
�
)Conv_layer_3/weights/truncated_normal/mulMul5Conv_layer_3/weights/truncated_normal/TruncatedNormal,Conv_layer_3/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(<
�
%Conv_layer_3/weights/truncated_normalAdd)Conv_layer_3/weights/truncated_normal/mul*Conv_layer_3/weights/truncated_normal/mean*
T0*&
_output_shapes
:(<
�
Conv_layer_3/weights/Variable
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *&
_output_shapes
:(<
�
$Conv_layer_3/weights/Variable/AssignAssignConv_layer_3/weights/Variable%Conv_layer_3/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
"Conv_layer_3/weights/Variable/readIdentityConv_layer_3/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
f
Conv_layer_3/biases/zerosConst*
valueB<*    *
dtype0*
_output_shapes
:<
�
Conv_layer_3/biases/Variable
VariableV2*
shape:<*
dtype0*
	container *
shared_name *
_output_shapes
:<
�
#Conv_layer_3/biases/Variable/AssignAssignConv_layer_3/biases/VariableConv_layer_3/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
!Conv_layer_3/biases/Variable/readIdentityConv_layer_3/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
Conv_layer_3/conv2dConv2DConv_layer_2/max-pooling"Conv_layer_3/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������

<
�
Conv_layer_3/addAddConv_layer_3/conv2d!Conv_layer_3/biases/Variable/read*
T0*/
_output_shapes
:���������

<
e
Conv_layer_3/reluReluConv_layer_3/add*
T0*/
_output_shapes
:���������

<
�
Conv_layer_3/max-poolingMaxPoolConv_layer_3/relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������<
�
+Conv_layer_4/weights/truncated_normal/shapeConst*%
valueB"      <   P   *
dtype0*
_output_shapes
:
o
*Conv_layer_4/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
q
,Conv_layer_4/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5Conv_layer_4/weights/truncated_normal/TruncatedNormalTruncatedNormal+Conv_layer_4/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:<P
�
)Conv_layer_4/weights/truncated_normal/mulMul5Conv_layer_4/weights/truncated_normal/TruncatedNormal,Conv_layer_4/weights/truncated_normal/stddev*
T0*&
_output_shapes
:<P
�
%Conv_layer_4/weights/truncated_normalAdd)Conv_layer_4/weights/truncated_normal/mul*Conv_layer_4/weights/truncated_normal/mean*
T0*&
_output_shapes
:<P
�
Conv_layer_4/weights/Variable
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *&
_output_shapes
:<P
�
$Conv_layer_4/weights/Variable/AssignAssignConv_layer_4/weights/Variable%Conv_layer_4/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
"Conv_layer_4/weights/Variable/readIdentityConv_layer_4/weights/Variable*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
f
Conv_layer_4/biases/zerosConst*
valueBP*    *
dtype0*
_output_shapes
:P
�
Conv_layer_4/biases/Variable
VariableV2*
shape:P*
dtype0*
	container *
shared_name *
_output_shapes
:P
�
#Conv_layer_4/biases/Variable/AssignAssignConv_layer_4/biases/VariableConv_layer_4/biases/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
!Conv_layer_4/biases/Variable/readIdentityConv_layer_4/biases/Variable*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
Conv_layer_4/conv2dConv2DConv_layer_3/max-pooling"Conv_layer_4/weights/Variable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������P
�
Conv_layer_4/addAddConv_layer_4/conv2d!Conv_layer_4/biases/Variable/read*
T0*/
_output_shapes
:���������P
e
Conv_layer_4/reluReluConv_layer_4/add*
T0*/
_output_shapes
:���������P
f
DeepID1/Reshape/shapeConst*
valueB"�����  *
dtype0*
_output_shapes
:
�
DeepID1/ReshapeReshapeConv_layer_3/max-poolingDeepID1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
h
DeepID1/Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
DeepID1/Reshape_1ReshapeConv_layer_4/reluDeepID1/Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������

w
&DeepID1/weights/truncated_normal/shapeConst*
valueB"�  �   *
dtype0*
_output_shapes
:
j
%DeepID1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
'DeepID1/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
0DeepID1/weights/truncated_normal/TruncatedNormalTruncatedNormal&DeepID1/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
��
�
$DeepID1/weights/truncated_normal/mulMul0DeepID1/weights/truncated_normal/TruncatedNormal'DeepID1/weights/truncated_normal/stddev*
T0* 
_output_shapes
:
��
�
 DeepID1/weights/truncated_normalAdd$DeepID1/weights/truncated_normal/mul%DeepID1/weights/truncated_normal/mean*
T0* 
_output_shapes
:
��
�
DeepID1/weights/Variable
VariableV2*
shape:
��*
dtype0*
	container *
shared_name * 
_output_shapes
:
��
�
DeepID1/weights/Variable/AssignAssignDeepID1/weights/Variable DeepID1/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/readIdentityDeepID1/weights/Variable*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
y
(DeepID1/weights_1/truncated_normal/shapeConst*
valueB"   �   *
dtype0*
_output_shapes
:
l
'DeepID1/weights_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
)DeepID1/weights_1/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
2DeepID1/weights_1/truncated_normal/TruncatedNormalTruncatedNormal(DeepID1/weights_1/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/truncated_normal/mulMul2DeepID1/weights_1/truncated_normal/TruncatedNormal)DeepID1/weights_1/truncated_normal/stddev*
T0* 
_output_shapes
:
�
�
�
"DeepID1/weights_1/truncated_normalAdd&DeepID1/weights_1/truncated_normal/mul'DeepID1/weights_1/truncated_normal/mean*
T0* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name * 
_output_shapes
:
�
�
�
!DeepID1/weights_1/Variable/AssignAssignDeepID1/weights_1/Variable"DeepID1/weights_1/truncated_normal*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable/readIdentityDeepID1/weights_1/Variable*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
c
DeepID1/biases/zerosConst*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DeepID1/biases/Variable
VariableV2*
shape:�*
dtype0*
	container *
shared_name *
_output_shapes	
:�
�
DeepID1/biases/Variable/AssignAssignDeepID1/biases/VariableDeepID1/biases/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/readIdentityDeepID1/biases/Variable*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/MatMulMatMulDeepID1/ReshapeDeepID1/weights/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
�
DeepID1/MatMul_1MatMulDeepID1/Reshape_1DeepID1/weights_1/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
g
DeepID1/addAddDeepID1/MatMulDeepID1/MatMul_1*
T0*(
_output_shapes
:����������
r
DeepID1/add_1AddDeepID1/addDeepID1/biases/Variable/read*
T0*(
_output_shapes
:����������
V
DeepID1/ReluReluDeepID1/add_1*
T0*(
_output_shapes
:����������
}
,loss/nn_layer/weights/truncated_normal/shapeConst*
valueB"�     *
dtype0*
_output_shapes
:
p
+loss/nn_layer/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
-loss/nn_layer/weights/truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
6loss/nn_layer/weights/truncated_normal/TruncatedNormalTruncatedNormal,loss/nn_layer/weights/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0* 
_output_shapes
:
��

�
*loss/nn_layer/weights/truncated_normal/mulMul6loss/nn_layer/weights/truncated_normal/TruncatedNormal-loss/nn_layer/weights/truncated_normal/stddev*
T0* 
_output_shapes
:
��

�
&loss/nn_layer/weights/truncated_normalAdd*loss/nn_layer/weights/truncated_normal/mul+loss/nn_layer/weights/truncated_normal/mean*
T0* 
_output_shapes
:
��

�
loss/nn_layer/weights/Variable
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name * 
_output_shapes
:
��

�
%loss/nn_layer/weights/Variable/AssignAssignloss/nn_layer/weights/Variable&loss/nn_layer/weights/truncated_normal*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
#loss/nn_layer/weights/Variable/readIdentityloss/nn_layer/weights/Variable*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

i
loss/nn_layer/biases/zerosConst*
valueB�
*    *
dtype0*
_output_shapes	
:�

�
loss/nn_layer/biases/Variable
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *
_output_shapes	
:�

�
$loss/nn_layer/biases/Variable/AssignAssignloss/nn_layer/biases/Variableloss/nn_layer/biases/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
"loss/nn_layer/biases/Variable/readIdentityloss/nn_layer/biases/Variable*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
loss/nn_layer/Wx_plus_b/MatMulMatMulDeepID1/Relu#loss/nn_layer/weights/Variable/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������

�
loss/nn_layer/Wx_plus_b/addAddloss/nn_layer/Wx_plus_b/MatMul"loss/nn_layer/biases/Variable/read*
T0*(
_output_shapes
:����������

K
	loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e

loss/ShapeShapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
M
loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
loss/Shape_1Shapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
L

loss/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
I
loss/SubSubloss/Rank_1
loss/Sub/y*
T0*
_output_shapes
: 
\
loss/Slice/beginPackloss/Sub*
N*
T0*

axis *
_output_shapes
:
Y
loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
v

loss/SliceSliceloss/Shape_1loss/Slice/beginloss/Slice/size*
T0*
Index0*
_output_shapes
:
g
loss/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
R
loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
loss/concatConcatV2loss/concat/values_0
loss/Sliceloss/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
�
loss/ReshapeReshapeloss/nn_layer/Wx_plus_b/addloss/concat*
T0*
Tshape0*0
_output_shapes
:������������������
M
loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
S
loss/Shape_2Shapeinput/y*
T0*
out_type0*
_output_shapes
:
N
loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
M

loss/Sub_1Subloss/Rank_2loss/Sub_1/y*
T0*
_output_shapes
: 
`
loss/Slice_1/beginPack
loss/Sub_1*
N*
T0*

axis *
_output_shapes
:
[
loss/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
|
loss/Slice_1Sliceloss/Shape_2loss/Slice_1/beginloss/Slice_1/size*
T0*
Index0*
_output_shapes
:
i
loss/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
T
loss/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
loss/concat_1ConcatV2loss/concat_1/values_0loss/Slice_1loss/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
z
loss/Reshape_1Reshapeinput/yloss/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
"loss/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsloss/Reshapeloss/Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
N
loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
K

loss/Sub_2Sub	loss/Rankloss/Sub_2/y*
T0*
_output_shapes
: 
\
loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
_
loss/Slice_2/sizePack
loss/Sub_2*
N*
T0*

axis *
_output_shapes
:
�
loss/Slice_2Slice
loss/Shapeloss/Slice_2/beginloss/Slice_2/size*
T0*
Index0*#
_output_shapes
:���������
�
loss/Reshape_2Reshape"loss/SoftmaxCrossEntropyWithLogitsloss/Slice_2*
T0*
Tshape0*#
_output_shapes
:���������
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
k
	loss/MeanMeanloss/Reshape_2
loss/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
X
loss/loss/tagsConst*
valueB B	loss/loss*
dtype0*
_output_shapes
: 
V
	loss/lossScalarSummaryloss/loss/tags	loss/Mean*
T0*
_output_shapes
: 
n
,accuracy/correct_prediction/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
"accuracy/correct_prediction/ArgMaxArgMaxloss/nn_layer/Wx_plus_b/add,accuracy/correct_prediction/ArgMax/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
p
.accuracy/correct_prediction/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
$accuracy/correct_prediction/ArgMax_1ArgMaxinput/y.accuracy/correct_prediction/ArgMax_1/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:���������
�
!accuracy/correct_prediction/EqualEqual"accuracy/correct_prediction/ArgMax$accuracy/correct_prediction/ArgMax_1*
T0	*#
_output_shapes
:���������
~
accuracy/accuracy/CastCast!accuracy/correct_prediction/Equal*

SrcT0
*

DstT0*#
_output_shapes
:���������
a
accuracy/accuracy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
accuracy/accuracy/MeanMeanaccuracy/accuracy/Castaccuracy/accuracy/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
l
accuracy/accuracy_1/tagsConst*$
valueB Baccuracy/accuracy_1*
dtype0*
_output_shapes
: 
w
accuracy/accuracy_1ScalarSummaryaccuracy/accuracy_1/tagsaccuracy/accuracy/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 
v
,train/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
&train/gradients/loss/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
r
$train/gradients/loss/Mean_grad/ShapeShapeloss/Reshape_2*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/TileTile&train/gradients/loss/Mean_grad/Reshape$train/gradients/loss/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
t
&train/gradients/loss/Mean_grad/Shape_1Shapeloss/Reshape_2*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
$train/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
:
�
#train/gradients/loss/Mean_grad/ProdProd&train/gradients/loss/Mean_grad/Shape_1$train/gradients/loss/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
:
�
%train/gradients/loss/Mean_grad/Prod_1Prod&train/gradients/loss/Mean_grad/Shape_2&train/gradients/loss/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
(train/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/MaximumMaximum%train/gradients/loss/Mean_grad/Prod_1(train/gradients/loss/Mean_grad/Maximum/y*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
'train/gradients/loss/Mean_grad/floordivFloorDiv#train/gradients/loss/Mean_grad/Prod&train/gradients/loss/Mean_grad/Maximum*
T0*9
_class/
-+loc:@train/gradients/loss/Mean_grad/Shape_1*
_output_shapes
: 
�
#train/gradients/loss/Mean_grad/CastCast'train/gradients/loss/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
&train/gradients/loss/Mean_grad/truedivRealDiv#train/gradients/loss/Mean_grad/Tile#train/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
)train/gradients/loss/Reshape_2_grad/ShapeShape"loss/SoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
+train/gradients/loss/Reshape_2_grad/ReshapeReshape&train/gradients/loss/Mean_grad/truediv)train/gradients/loss/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
train/gradients/zeros_like	ZerosLike$loss/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
Ftrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Btrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims+train/gradients/loss/Reshape_2_grad/ReshapeFtrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:���������
�
;train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mulMulBtrain/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/ExpandDims$loss/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
'train/gradients/loss/Reshape_grad/ShapeShapeloss/nn_layer/Wx_plus_b/add*
T0*
out_type0*
_output_shapes
:
�
)train/gradients/loss/Reshape_grad/ReshapeReshape;train/gradients/loss/SoftmaxCrossEntropyWithLogits_grad/mul'train/gradients/loss/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������

�
6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ShapeShapeloss/nn_layer/Wx_plus_b/MatMul*
T0*
out_type0*
_output_shapes
:
�
8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1Const*
valueB:�
*
dtype0*
_output_shapes
:
�
Ftrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgsBroadcastGradientArgs6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
4train/gradients/loss/nn_layer/Wx_plus_b/add_grad/SumSum)train/gradients/loss/Reshape_grad/ReshapeFtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ReshapeReshape4train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������

�
6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum_1Sum)train/gradients/loss/Reshape_grad/ReshapeHtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
:train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1Reshape6train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Sum_18train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�

�
Atrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_depsNoOp9^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape;^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1
�
Itrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependencyIdentity8train/gradients/loss/nn_layer/Wx_plus_b/add_grad/ReshapeB^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape*(
_output_shapes
:����������

�
Ktrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency_1Identity:train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1B^train/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/loss/nn_layer/Wx_plus_b/add_grad/Reshape_1*
_output_shapes	
:�

�
:train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMulMatMulItrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency#loss/nn_layer/weights/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
<train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1MatMulDeepID1/ReluItrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��

�
Dtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_depsNoOp;^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul=^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1
�
Ltrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependencyIdentity:train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMulE^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ntrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependency_1Identity<train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1E^train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/MatMul_1* 
_output_shapes
:
��

�
*train/gradients/DeepID1/Relu_grad/ReluGradReluGradLtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependencyDeepID1/Relu*
T0*(
_output_shapes
:����������
s
(train/gradients/DeepID1/add_1_grad/ShapeShapeDeepID1/add*
T0*
out_type0*
_output_shapes
:
u
*train/gradients/DeepID1/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
8train/gradients/DeepID1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs(train/gradients/DeepID1/add_1_grad/Shape*train/gradients/DeepID1/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&train/gradients/DeepID1/add_1_grad/SumSum*train/gradients/DeepID1/Relu_grad/ReluGrad8train/gradients/DeepID1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
*train/gradients/DeepID1/add_1_grad/ReshapeReshape&train/gradients/DeepID1/add_1_grad/Sum(train/gradients/DeepID1/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
(train/gradients/DeepID1/add_1_grad/Sum_1Sum*train/gradients/DeepID1/Relu_grad/ReluGrad:train/gradients/DeepID1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
,train/gradients/DeepID1/add_1_grad/Reshape_1Reshape(train/gradients/DeepID1/add_1_grad/Sum_1*train/gradients/DeepID1/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
3train/gradients/DeepID1/add_1_grad/tuple/group_depsNoOp+^train/gradients/DeepID1/add_1_grad/Reshape-^train/gradients/DeepID1/add_1_grad/Reshape_1
�
;train/gradients/DeepID1/add_1_grad/tuple/control_dependencyIdentity*train/gradients/DeepID1/add_1_grad/Reshape4^train/gradients/DeepID1/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/add_1_grad/Reshape*(
_output_shapes
:����������
�
=train/gradients/DeepID1/add_1_grad/tuple/control_dependency_1Identity,train/gradients/DeepID1/add_1_grad/Reshape_14^train/gradients/DeepID1/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/add_1_grad/Reshape_1*
_output_shapes	
:�
t
&train/gradients/DeepID1/add_grad/ShapeShapeDeepID1/MatMul*
T0*
out_type0*
_output_shapes
:
x
(train/gradients/DeepID1/add_grad/Shape_1ShapeDeepID1/MatMul_1*
T0*
out_type0*
_output_shapes
:
�
6train/gradients/DeepID1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/DeepID1/add_grad/Shape(train/gradients/DeepID1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/DeepID1/add_grad/SumSum;train/gradients/DeepID1/add_1_grad/tuple/control_dependency6train/gradients/DeepID1/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
(train/gradients/DeepID1/add_grad/ReshapeReshape$train/gradients/DeepID1/add_grad/Sum&train/gradients/DeepID1/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
&train/gradients/DeepID1/add_grad/Sum_1Sum;train/gradients/DeepID1/add_1_grad/tuple/control_dependency8train/gradients/DeepID1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
*train/gradients/DeepID1/add_grad/Reshape_1Reshape&train/gradients/DeepID1/add_grad/Sum_1(train/gradients/DeepID1/add_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
1train/gradients/DeepID1/add_grad/tuple/group_depsNoOp)^train/gradients/DeepID1/add_grad/Reshape+^train/gradients/DeepID1/add_grad/Reshape_1
�
9train/gradients/DeepID1/add_grad/tuple/control_dependencyIdentity(train/gradients/DeepID1/add_grad/Reshape2^train/gradients/DeepID1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/DeepID1/add_grad/Reshape*(
_output_shapes
:����������
�
;train/gradients/DeepID1/add_grad/tuple/control_dependency_1Identity*train/gradients/DeepID1/add_grad/Reshape_12^train/gradients/DeepID1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/add_grad/Reshape_1*(
_output_shapes
:����������
�
*train/gradients/DeepID1/MatMul_grad/MatMulMatMul9train/gradients/DeepID1/add_grad/tuple/control_dependencyDeepID1/weights/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������
�
,train/gradients/DeepID1/MatMul_grad/MatMul_1MatMulDeepID1/Reshape9train/gradients/DeepID1/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
��
�
4train/gradients/DeepID1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/DeepID1/MatMul_grad/MatMul-^train/gradients/DeepID1/MatMul_grad/MatMul_1
�
<train/gradients/DeepID1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/DeepID1/MatMul_grad/MatMul5^train/gradients/DeepID1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/DeepID1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/DeepID1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/DeepID1/MatMul_grad/MatMul_15^train/gradients/DeepID1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
,train/gradients/DeepID1/MatMul_1_grad/MatMulMatMul;train/gradients/DeepID1/add_grad/tuple/control_dependency_1DeepID1/weights_1/Variable/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:����������

�
.train/gradients/DeepID1/MatMul_1_grad/MatMul_1MatMulDeepID1/Reshape_1;train/gradients/DeepID1/add_grad/tuple/control_dependency_1*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
�
�
�
6train/gradients/DeepID1/MatMul_1_grad/tuple/group_depsNoOp-^train/gradients/DeepID1/MatMul_1_grad/MatMul/^train/gradients/DeepID1/MatMul_1_grad/MatMul_1
�
>train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependencyIdentity,train/gradients/DeepID1/MatMul_1_grad/MatMul7^train/gradients/DeepID1/MatMul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/DeepID1/MatMul_1_grad/MatMul*(
_output_shapes
:����������

�
@train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency_1Identity.train/gradients/DeepID1/MatMul_1_grad/MatMul_17^train/gradients/DeepID1/MatMul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@train/gradients/DeepID1/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�
�
�
*train/gradients/DeepID1/Reshape_grad/ShapeShapeConv_layer_3/max-pooling*
T0*
out_type0*
_output_shapes
:
�
,train/gradients/DeepID1/Reshape_grad/ReshapeReshape<train/gradients/DeepID1/MatMul_grad/tuple/control_dependency*train/gradients/DeepID1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������<
}
,train/gradients/DeepID1/Reshape_1_grad/ShapeShapeConv_layer_4/relu*
T0*
out_type0*
_output_shapes
:
�
.train/gradients/DeepID1/Reshape_1_grad/ReshapeReshape>train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency,train/gradients/DeepID1/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������P
�
/train/gradients/Conv_layer_4/relu_grad/ReluGradReluGrad.train/gradients/DeepID1/Reshape_1_grad/ReshapeConv_layer_4/relu*
T0*/
_output_shapes
:���������P
~
+train/gradients/Conv_layer_4/add_grad/ShapeShapeConv_layer_4/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_4/add_grad/Shape_1Const*
valueB:P*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_4/add_grad/Shape-train/gradients/Conv_layer_4/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_4/add_grad/SumSum/train/gradients/Conv_layer_4/relu_grad/ReluGrad;train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_4/add_grad/ReshapeReshape)train/gradients/Conv_layer_4/add_grad/Sum+train/gradients/Conv_layer_4/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������P
�
+train/gradients/Conv_layer_4/add_grad/Sum_1Sum/train/gradients/Conv_layer_4/relu_grad/ReluGrad=train/gradients/Conv_layer_4/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_4/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_4/add_grad/Sum_1-train/gradients/Conv_layer_4/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:P
�
6train/gradients/Conv_layer_4/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_4/add_grad/Reshape0^train/gradients/Conv_layer_4/add_grad/Reshape_1
�
>train/gradients/Conv_layer_4/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_4/add_grad/Reshape7^train/gradients/Conv_layer_4/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_4/add_grad/Reshape*/
_output_shapes
:���������P
�
@train/gradients/Conv_layer_4/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_4/add_grad/Reshape_17^train/gradients/Conv_layer_4/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_4/add_grad/Reshape_1*
_output_shapes
:P
�
/train/gradients/Conv_layer_4/conv2d_grad/ShapeNShapeNConv_layer_3/max-pooling"Conv_layer_4/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_4/conv2d_grad/ShapeN"Conv_layer_4/weights/Variable/read>train/gradients/Conv_layer_4/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_3/max-pooling1train/gradients/Conv_layer_4/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_4/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_4/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_4/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������<
�
Ctrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_4/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_4/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:<P
�
train/gradients/AddNAddN,train/gradients/DeepID1/Reshape_grad/ReshapeAtrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency*
N*
T0*?
_class5
31loc:@train/gradients/DeepID1/Reshape_grad/Reshape*/
_output_shapes
:���������<
�
9train/gradients/Conv_layer_3/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_3/reluConv_layer_3/max-poolingtrain/gradients/AddN*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������

<
�
/train/gradients/Conv_layer_3/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_3/max-pooling_grad/MaxPoolGradConv_layer_3/relu*
T0*/
_output_shapes
:���������

<
~
+train/gradients/Conv_layer_3/add_grad/ShapeShapeConv_layer_3/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_3/add_grad/Shape_1Const*
valueB:<*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_3/add_grad/Shape-train/gradients/Conv_layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_3/add_grad/SumSum/train/gradients/Conv_layer_3/relu_grad/ReluGrad;train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_3/add_grad/ReshapeReshape)train/gradients/Conv_layer_3/add_grad/Sum+train/gradients/Conv_layer_3/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������

<
�
+train/gradients/Conv_layer_3/add_grad/Sum_1Sum/train/gradients/Conv_layer_3/relu_grad/ReluGrad=train/gradients/Conv_layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_3/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_3/add_grad/Sum_1-train/gradients/Conv_layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:<
�
6train/gradients/Conv_layer_3/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_3/add_grad/Reshape0^train/gradients/Conv_layer_3/add_grad/Reshape_1
�
>train/gradients/Conv_layer_3/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_3/add_grad/Reshape7^train/gradients/Conv_layer_3/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_3/add_grad/Reshape*/
_output_shapes
:���������

<
�
@train/gradients/Conv_layer_3/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_3/add_grad/Reshape_17^train/gradients/Conv_layer_3/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_3/add_grad/Reshape_1*
_output_shapes
:<
�
/train/gradients/Conv_layer_3/conv2d_grad/ShapeNShapeNConv_layer_2/max-pooling"Conv_layer_3/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_3/conv2d_grad/ShapeN"Conv_layer_3/weights/Variable/read>train/gradients/Conv_layer_3/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_2/max-pooling1train/gradients/Conv_layer_3/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_3/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_3/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_3/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������(
�
Ctrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_3/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_3/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:(<
�
9train/gradients/Conv_layer_2/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_2/reluConv_layer_2/max-poolingAtrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������(
�
/train/gradients/Conv_layer_2/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_2/max-pooling_grad/MaxPoolGradConv_layer_2/relu*
T0*/
_output_shapes
:���������(
~
+train/gradients/Conv_layer_2/add_grad/ShapeShapeConv_layer_2/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_2/add_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_2/add_grad/Shape-train/gradients/Conv_layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_2/add_grad/SumSum/train/gradients/Conv_layer_2/relu_grad/ReluGrad;train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_2/add_grad/ReshapeReshape)train/gradients/Conv_layer_2/add_grad/Sum+train/gradients/Conv_layer_2/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������(
�
+train/gradients/Conv_layer_2/add_grad/Sum_1Sum/train/gradients/Conv_layer_2/relu_grad/ReluGrad=train/gradients/Conv_layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_2/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_2/add_grad/Sum_1-train/gradients/Conv_layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:(
�
6train/gradients/Conv_layer_2/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_2/add_grad/Reshape0^train/gradients/Conv_layer_2/add_grad/Reshape_1
�
>train/gradients/Conv_layer_2/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_2/add_grad/Reshape7^train/gradients/Conv_layer_2/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_2/add_grad/Reshape*/
_output_shapes
:���������(
�
@train/gradients/Conv_layer_2/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_2/add_grad/Reshape_17^train/gradients/Conv_layer_2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_2/add_grad/Reshape_1*
_output_shapes
:(
�
/train/gradients/Conv_layer_2/conv2d_grad/ShapeNShapeNConv_layer_1/max-pooling"Conv_layer_2/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_2/conv2d_grad/ShapeN"Conv_layer_2/weights/Variable/read>train/gradients/Conv_layer_2/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_layer_1/max-pooling1train/gradients/Conv_layer_2/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_2/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_2/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_2/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Ctrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_2/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_2/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:(
�
9train/gradients/Conv_layer_1/max-pooling_grad/MaxPoolGradMaxPoolGradConv_layer_1/reluConv_layer_1/max-poolingAtrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*
T0*/
_output_shapes
:���������44
�
/train/gradients/Conv_layer_1/relu_grad/ReluGradReluGrad9train/gradients/Conv_layer_1/max-pooling_grad/MaxPoolGradConv_layer_1/relu*
T0*/
_output_shapes
:���������44
~
+train/gradients/Conv_layer_1/add_grad/ShapeShapeConv_layer_1/conv2d*
T0*
out_type0*
_output_shapes
:
w
-train/gradients/Conv_layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
;train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs+train/gradients/Conv_layer_1/add_grad/Shape-train/gradients/Conv_layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
)train/gradients/Conv_layer_1/add_grad/SumSum/train/gradients/Conv_layer_1/relu_grad/ReluGrad;train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
-train/gradients/Conv_layer_1/add_grad/ReshapeReshape)train/gradients/Conv_layer_1/add_grad/Sum+train/gradients/Conv_layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������44
�
+train/gradients/Conv_layer_1/add_grad/Sum_1Sum/train/gradients/Conv_layer_1/relu_grad/ReluGrad=train/gradients/Conv_layer_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
�
/train/gradients/Conv_layer_1/add_grad/Reshape_1Reshape+train/gradients/Conv_layer_1/add_grad/Sum_1-train/gradients/Conv_layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
6train/gradients/Conv_layer_1/add_grad/tuple/group_depsNoOp.^train/gradients/Conv_layer_1/add_grad/Reshape0^train/gradients/Conv_layer_1/add_grad/Reshape_1
�
>train/gradients/Conv_layer_1/add_grad/tuple/control_dependencyIdentity-train/gradients/Conv_layer_1/add_grad/Reshape7^train/gradients/Conv_layer_1/add_grad/tuple/group_deps*
T0*@
_class6
42loc:@train/gradients/Conv_layer_1/add_grad/Reshape*/
_output_shapes
:���������44
�
@train/gradients/Conv_layer_1/add_grad/tuple/control_dependency_1Identity/train/gradients/Conv_layer_1/add_grad/Reshape_17^train/gradients/Conv_layer_1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/Conv_layer_1/add_grad/Reshape_1*
_output_shapes
:
�
/train/gradients/Conv_layer_1/conv2d_grad/ShapeNShapeNinput/x"Conv_layer_1/weights/Variable/read*
N*
T0*
out_type0* 
_output_shapes
::
�
<train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInputConv2DBackpropInput/train/gradients/Conv_layer_1/conv2d_grad/ShapeN"Conv_layer_1/weights/Variable/read>train/gradients/Conv_layer_1/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
=train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/x1train/gradients/Conv_layer_1/conv2d_grad/ShapeN:1>train/gradients/Conv_layer_1/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4������������������������������������
�
9train/gradients/Conv_layer_1/conv2d_grad/tuple/group_depsNoOp=^train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput>^train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter
�
Atrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependencyIdentity<train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput:^train/gradients/Conv_layer_1/conv2d_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropInput*/
_output_shapes
:���������77
�
Ctrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependency_1Identity=train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter:^train/gradients/Conv_layer_1/conv2d_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/Conv_layer_1/conv2d_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
train/beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta1_power/readIdentitytrain/beta1_power*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/readIdentitytrain/beta2_power*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
4Conv_layer_1/weights/Variable/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
"Conv_layer_1/weights/Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
)Conv_layer_1/weights/Variable/Adam/AssignAssign"Conv_layer_1/weights/Variable/Adam4Conv_layer_1/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
'Conv_layer_1/weights/Variable/Adam/readIdentity"Conv_layer_1/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
6Conv_layer_1/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
$Conv_layer_1/weights/Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
+Conv_layer_1/weights/Variable/Adam_1/AssignAssign$Conv_layer_1/weights/Variable/Adam_16Conv_layer_1/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
)Conv_layer_1/weights/Variable/Adam_1/readIdentity$Conv_layer_1/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
3Conv_layer_1/biases/Variable/Adam/Initializer/zerosConst*
valueB*    *
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
!Conv_layer_1/biases/Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
(Conv_layer_1/biases/Variable/Adam/AssignAssign!Conv_layer_1/biases/Variable/Adam3Conv_layer_1/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
&Conv_layer_1/biases/Variable/Adam/readIdentity!Conv_layer_1/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
5Conv_layer_1/biases/Variable/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
#Conv_layer_1/biases/Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
*Conv_layer_1/biases/Variable/Adam_1/AssignAssign#Conv_layer_1/biases/Variable/Adam_15Conv_layer_1/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
(Conv_layer_1/biases/Variable/Adam_1/readIdentity#Conv_layer_1/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
4Conv_layer_2/weights/Variable/Adam/Initializer/zerosConst*%
valueB(*    *
dtype0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
"Conv_layer_2/weights/Variable/Adam
VariableV2*
shape:(*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
)Conv_layer_2/weights/Variable/Adam/AssignAssign"Conv_layer_2/weights/Variable/Adam4Conv_layer_2/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
'Conv_layer_2/weights/Variable/Adam/readIdentity"Conv_layer_2/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
6Conv_layer_2/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB(*    *
dtype0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
$Conv_layer_2/weights/Variable/Adam_1
VariableV2*
shape:(*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
+Conv_layer_2/weights/Variable/Adam_1/AssignAssign$Conv_layer_2/weights/Variable/Adam_16Conv_layer_2/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
)Conv_layer_2/weights/Variable/Adam_1/readIdentity$Conv_layer_2/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
3Conv_layer_2/biases/Variable/Adam/Initializer/zerosConst*
valueB(*    *
dtype0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
!Conv_layer_2/biases/Variable/Adam
VariableV2*
shape:(*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
(Conv_layer_2/biases/Variable/Adam/AssignAssign!Conv_layer_2/biases/Variable/Adam3Conv_layer_2/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
&Conv_layer_2/biases/Variable/Adam/readIdentity!Conv_layer_2/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
5Conv_layer_2/biases/Variable/Adam_1/Initializer/zerosConst*
valueB(*    *
dtype0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
#Conv_layer_2/biases/Variable/Adam_1
VariableV2*
shape:(*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
*Conv_layer_2/biases/Variable/Adam_1/AssignAssign#Conv_layer_2/biases/Variable/Adam_15Conv_layer_2/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
(Conv_layer_2/biases/Variable/Adam_1/readIdentity#Conv_layer_2/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
4Conv_layer_3/weights/Variable/Adam/Initializer/zerosConst*%
valueB(<*    *
dtype0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
"Conv_layer_3/weights/Variable/Adam
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
)Conv_layer_3/weights/Variable/Adam/AssignAssign"Conv_layer_3/weights/Variable/Adam4Conv_layer_3/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
'Conv_layer_3/weights/Variable/Adam/readIdentity"Conv_layer_3/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
6Conv_layer_3/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB(<*    *
dtype0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
$Conv_layer_3/weights/Variable/Adam_1
VariableV2*
shape:(<*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
+Conv_layer_3/weights/Variable/Adam_1/AssignAssign$Conv_layer_3/weights/Variable/Adam_16Conv_layer_3/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
)Conv_layer_3/weights/Variable/Adam_1/readIdentity$Conv_layer_3/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
3Conv_layer_3/biases/Variable/Adam/Initializer/zerosConst*
valueB<*    *
dtype0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
!Conv_layer_3/biases/Variable/Adam
VariableV2*
shape:<*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
(Conv_layer_3/biases/Variable/Adam/AssignAssign!Conv_layer_3/biases/Variable/Adam3Conv_layer_3/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
&Conv_layer_3/biases/Variable/Adam/readIdentity!Conv_layer_3/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
5Conv_layer_3/biases/Variable/Adam_1/Initializer/zerosConst*
valueB<*    *
dtype0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
#Conv_layer_3/biases/Variable/Adam_1
VariableV2*
shape:<*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
*Conv_layer_3/biases/Variable/Adam_1/AssignAssign#Conv_layer_3/biases/Variable/Adam_15Conv_layer_3/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
(Conv_layer_3/biases/Variable/Adam_1/readIdentity#Conv_layer_3/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
4Conv_layer_4/weights/Variable/Adam/Initializer/zerosConst*%
valueB<P*    *
dtype0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
"Conv_layer_4/weights/Variable/Adam
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
)Conv_layer_4/weights/Variable/Adam/AssignAssign"Conv_layer_4/weights/Variable/Adam4Conv_layer_4/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
'Conv_layer_4/weights/Variable/Adam/readIdentity"Conv_layer_4/weights/Variable/Adam*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
6Conv_layer_4/weights/Variable/Adam_1/Initializer/zerosConst*%
valueB<P*    *
dtype0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
$Conv_layer_4/weights/Variable/Adam_1
VariableV2*
shape:<P*
dtype0*
	container *
shared_name *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
+Conv_layer_4/weights/Variable/Adam_1/AssignAssign$Conv_layer_4/weights/Variable/Adam_16Conv_layer_4/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
)Conv_layer_4/weights/Variable/Adam_1/readIdentity$Conv_layer_4/weights/Variable/Adam_1*
T0*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
3Conv_layer_4/biases/Variable/Adam/Initializer/zerosConst*
valueBP*    *
dtype0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
!Conv_layer_4/biases/Variable/Adam
VariableV2*
shape:P*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
(Conv_layer_4/biases/Variable/Adam/AssignAssign!Conv_layer_4/biases/Variable/Adam3Conv_layer_4/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
&Conv_layer_4/biases/Variable/Adam/readIdentity!Conv_layer_4/biases/Variable/Adam*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
5Conv_layer_4/biases/Variable/Adam_1/Initializer/zerosConst*
valueBP*    *
dtype0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
#Conv_layer_4/biases/Variable/Adam_1
VariableV2*
shape:P*
dtype0*
	container *
shared_name */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
*Conv_layer_4/biases/Variable/Adam_1/AssignAssign#Conv_layer_4/biases/Variable/Adam_15Conv_layer_4/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
(Conv_layer_4/biases/Variable/Adam_1/readIdentity#Conv_layer_4/biases/Variable/Adam_1*
T0*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
/DeepID1/weights/Variable/Adam/Initializer/zerosConst*
valueB
��*    *
dtype0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/Adam
VariableV2*
shape:
��*
dtype0*
	container *
shared_name *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
$DeepID1/weights/Variable/Adam/AssignAssignDeepID1/weights/Variable/Adam/DeepID1/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
"DeepID1/weights/Variable/Adam/readIdentityDeepID1/weights/Variable/Adam*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
1DeepID1/weights/Variable/Adam_1/Initializer/zerosConst*
valueB
��*    *
dtype0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
DeepID1/weights/Variable/Adam_1
VariableV2*
shape:
��*
dtype0*
	container *
shared_name *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
&DeepID1/weights/Variable/Adam_1/AssignAssignDeepID1/weights/Variable/Adam_11DeepID1/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
$DeepID1/weights/Variable/Adam_1/readIdentityDeepID1/weights/Variable/Adam_1*
T0*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
1DeepID1/weights_1/Variable/Adam/Initializer/zerosConst*
valueB
�
�*    *
dtype0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
DeepID1/weights_1/Variable/Adam
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/Variable/Adam/AssignAssignDeepID1/weights_1/Variable/Adam1DeepID1/weights_1/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
$DeepID1/weights_1/Variable/Adam/readIdentityDeepID1/weights_1/Variable/Adam*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
3DeepID1/weights_1/Variable/Adam_1/Initializer/zerosConst*
valueB
�
�*    *
dtype0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
!DeepID1/weights_1/Variable/Adam_1
VariableV2*
shape:
�
�*
dtype0*
	container *
shared_name *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
(DeepID1/weights_1/Variable/Adam_1/AssignAssign!DeepID1/weights_1/Variable/Adam_13DeepID1/weights_1/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
&DeepID1/weights_1/Variable/Adam_1/readIdentity!DeepID1/weights_1/Variable/Adam_1*
T0*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
.DeepID1/biases/Variable/Adam/Initializer/zerosConst*
valueB�*    *
dtype0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/Adam
VariableV2*
shape:�*
dtype0*
	container *
shared_name **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
#DeepID1/biases/Variable/Adam/AssignAssignDeepID1/biases/Variable/Adam.DeepID1/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
!DeepID1/biases/Variable/Adam/readIdentityDeepID1/biases/Variable/Adam*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
0DeepID1/biases/Variable/Adam_1/Initializer/zerosConst*
valueB�*    *
dtype0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
DeepID1/biases/Variable/Adam_1
VariableV2*
shape:�*
dtype0*
	container *
shared_name **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
%DeepID1/biases/Variable/Adam_1/AssignAssignDeepID1/biases/Variable/Adam_10DeepID1/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
#DeepID1/biases/Variable/Adam_1/readIdentityDeepID1/biases/Variable/Adam_1*
T0**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
5loss/nn_layer/weights/Variable/Adam/Initializer/zerosConst*
valueB
��
*    *
dtype0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
#loss/nn_layer/weights/Variable/Adam
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
*loss/nn_layer/weights/Variable/Adam/AssignAssign#loss/nn_layer/weights/Variable/Adam5loss/nn_layer/weights/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
(loss/nn_layer/weights/Variable/Adam/readIdentity#loss/nn_layer/weights/Variable/Adam*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
7loss/nn_layer/weights/Variable/Adam_1/Initializer/zerosConst*
valueB
��
*    *
dtype0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
%loss/nn_layer/weights/Variable/Adam_1
VariableV2*
shape:
��
*
dtype0*
	container *
shared_name *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
,loss/nn_layer/weights/Variable/Adam_1/AssignAssign%loss/nn_layer/weights/Variable/Adam_17loss/nn_layer/weights/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
*loss/nn_layer/weights/Variable/Adam_1/readIdentity%loss/nn_layer/weights/Variable/Adam_1*
T0*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
4loss/nn_layer/biases/Variable/Adam/Initializer/zerosConst*
valueB�
*    *
dtype0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
"loss/nn_layer/biases/Variable/Adam
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
)loss/nn_layer/biases/Variable/Adam/AssignAssign"loss/nn_layer/biases/Variable/Adam4loss/nn_layer/biases/Variable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
'loss/nn_layer/biases/Variable/Adam/readIdentity"loss/nn_layer/biases/Variable/Adam*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
6loss/nn_layer/biases/Variable/Adam_1/Initializer/zerosConst*
valueB�
*    *
dtype0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
$loss/nn_layer/biases/Variable/Adam_1
VariableV2*
shape:�
*
dtype0*
	container *
shared_name *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
+loss/nn_layer/biases/Variable/Adam_1/AssignAssign$loss/nn_layer/biases/Variable/Adam_16loss/nn_layer/biases/Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
)loss/nn_layer/biases/Variable/Adam_1/readIdentity$loss/nn_layer/biases/Variable/Adam_1*
T0*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

]
train/Adam/learning_rateConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
9train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam	ApplyAdamConv_layer_1/weights/Variable"Conv_layer_1/weights/Variable/Adam$Conv_layer_1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_1/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
8train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam	ApplyAdamConv_layer_1/biases/Variable!Conv_layer_1/biases/Variable/Adam#Conv_layer_1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_1/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
9train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam	ApplyAdamConv_layer_2/weights/Variable"Conv_layer_2/weights/Variable/Adam$Conv_layer_2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_2/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
8train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam	ApplyAdamConv_layer_2/biases/Variable!Conv_layer_2/biases/Variable/Adam#Conv_layer_2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_2/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
9train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam	ApplyAdamConv_layer_3/weights/Variable"Conv_layer_3/weights/Variable/Adam$Conv_layer_3/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_3/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
8train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam	ApplyAdamConv_layer_3/biases/Variable!Conv_layer_3/biases/Variable/Adam#Conv_layer_3/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_3/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
9train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam	ApplyAdamConv_layer_4/weights/Variable"Conv_layer_4/weights/Variable/Adam$Conv_layer_4/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonCtrain/gradients/Conv_layer_4/conv2d_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
8train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam	ApplyAdamConv_layer_4/biases/Variable!Conv_layer_4/biases/Variable/Adam#Conv_layer_4/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/Conv_layer_4/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( */
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
4train/Adam/update_DeepID1/weights/Variable/ApplyAdam	ApplyAdamDeepID1/weights/VariableDeepID1/weights/Variable/AdamDeepID1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/DeepID1/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
6train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam	ApplyAdamDeepID1/weights_1/VariableDeepID1/weights_1/Variable/Adam!DeepID1/weights_1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon@train/gradients/DeepID1/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
3train/Adam/update_DeepID1/biases/Variable/ApplyAdam	ApplyAdamDeepID1/biases/VariableDeepID1/biases/Variable/AdamDeepID1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/DeepID1/add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( **
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
:train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam	ApplyAdamloss/nn_layer/weights/Variable#loss/nn_layer/weights/Variable/Adam%loss/nn_layer/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/loss/nn_layer/Wx_plus_b/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
9train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam	ApplyAdamloss/nn_layer/biases/Variable"loss/nn_layer/biases/Variable/Adam$loss/nn_layer/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonKtrain/gradients/loss/nn_layer/Wx_plus_b/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*
validate_shape(*
use_locking( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam*
T0*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*
validate_shape(*
use_locking( */
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�

train/AdamNoOp:^train/Adam/update_Conv_layer_1/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_1/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_2/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_2/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_3/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_3/biases/Variable/ApplyAdam:^train/Adam/update_Conv_layer_4/weights/Variable/ApplyAdam9^train/Adam/update_Conv_layer_4/biases/Variable/ApplyAdam5^train/Adam/update_DeepID1/weights/Variable/ApplyAdam7^train/Adam/update_DeepID1/weights_1/Variable/ApplyAdam4^train/Adam/update_DeepID1/biases/Variable/ApplyAdam;^train/Adam/update_loss/nn_layer/weights/Variable/ApplyAdam:^train/Adam/update_loss/nn_layer/biases/Variable/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
c
Merge/MergeSummaryMergeSummary	loss/lossaccuracy/accuracy_1*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�

value�
B�
)BConv_layer_1/biases/VariableB!Conv_layer_1/biases/Variable/AdamB#Conv_layer_1/biases/Variable/Adam_1BConv_layer_1/weights/VariableB"Conv_layer_1/weights/Variable/AdamB$Conv_layer_1/weights/Variable/Adam_1BConv_layer_2/biases/VariableB!Conv_layer_2/biases/Variable/AdamB#Conv_layer_2/biases/Variable/Adam_1BConv_layer_2/weights/VariableB"Conv_layer_2/weights/Variable/AdamB$Conv_layer_2/weights/Variable/Adam_1BConv_layer_3/biases/VariableB!Conv_layer_3/biases/Variable/AdamB#Conv_layer_3/biases/Variable/Adam_1BConv_layer_3/weights/VariableB"Conv_layer_3/weights/Variable/AdamB$Conv_layer_3/weights/Variable/Adam_1BConv_layer_4/biases/VariableB!Conv_layer_4/biases/Variable/AdamB#Conv_layer_4/biases/Variable/Adam_1BConv_layer_4/weights/VariableB"Conv_layer_4/weights/Variable/AdamB$Conv_layer_4/weights/Variable/Adam_1BDeepID1/biases/VariableBDeepID1/biases/Variable/AdamBDeepID1/biases/Variable/Adam_1BDeepID1/weights/VariableBDeepID1/weights/Variable/AdamBDeepID1/weights/Variable/Adam_1BDeepID1/weights_1/VariableBDeepID1/weights_1/Variable/AdamB!DeepID1/weights_1/Variable/Adam_1Bloss/nn_layer/biases/VariableB"loss/nn_layer/biases/Variable/AdamB$loss/nn_layer/biases/Variable/Adam_1Bloss/nn_layer/weights/VariableB#loss/nn_layer/weights/Variable/AdamB%loss/nn_layer/weights/Variable/Adam_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:)
�
save/SaveV2/shape_and_slicesConst*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:)
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConv_layer_1/biases/Variable!Conv_layer_1/biases/Variable/Adam#Conv_layer_1/biases/Variable/Adam_1Conv_layer_1/weights/Variable"Conv_layer_1/weights/Variable/Adam$Conv_layer_1/weights/Variable/Adam_1Conv_layer_2/biases/Variable!Conv_layer_2/biases/Variable/Adam#Conv_layer_2/biases/Variable/Adam_1Conv_layer_2/weights/Variable"Conv_layer_2/weights/Variable/Adam$Conv_layer_2/weights/Variable/Adam_1Conv_layer_3/biases/Variable!Conv_layer_3/biases/Variable/Adam#Conv_layer_3/biases/Variable/Adam_1Conv_layer_3/weights/Variable"Conv_layer_3/weights/Variable/Adam$Conv_layer_3/weights/Variable/Adam_1Conv_layer_4/biases/Variable!Conv_layer_4/biases/Variable/Adam#Conv_layer_4/biases/Variable/Adam_1Conv_layer_4/weights/Variable"Conv_layer_4/weights/Variable/Adam$Conv_layer_4/weights/Variable/Adam_1DeepID1/biases/VariableDeepID1/biases/Variable/AdamDeepID1/biases/Variable/Adam_1DeepID1/weights/VariableDeepID1/weights/Variable/AdamDeepID1/weights/Variable/Adam_1DeepID1/weights_1/VariableDeepID1/weights_1/Variable/Adam!DeepID1/weights_1/Variable/Adam_1loss/nn_layer/biases/Variable"loss/nn_layer/biases/Variable/Adam$loss/nn_layer/biases/Variable/Adam_1loss/nn_layer/weights/Variable#loss/nn_layer/weights/Variable/Adam%loss/nn_layer/weights/Variable/Adam_1train/beta1_powertrain/beta2_power*7
dtypes-
+2)
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*1
value(B&BConv_layer_1/biases/Variable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignConv_layer_1/biases/Variablesave/RestoreV2*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_1/tensor_namesConst*6
value-B+B!Conv_layer_1/biases/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign!Conv_layer_1/biases/Variable/Adamsave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_2/tensor_namesConst*8
value/B-B#Conv_layer_1/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign#Conv_layer_1/biases/Variable/Adam_1save/RestoreV2_2*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
:
�
save/RestoreV2_3/tensor_namesConst*2
value)B'BConv_layer_1/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3AssignConv_layer_1/weights/Variablesave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_4/tensor_namesConst*7
value.B,B"Conv_layer_1/weights/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign"Conv_layer_1/weights/Variable/Adamsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_5/tensor_namesConst*9
value0B.B$Conv_layer_1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign$Conv_layer_1/weights/Variable/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_1/weights/Variable*&
_output_shapes
:
�
save/RestoreV2_6/tensor_namesConst*1
value(B&BConv_layer_2/biases/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6AssignConv_layer_2/biases/Variablesave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_7/tensor_namesConst*6
value-B+B!Conv_layer_2/biases/Variable/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assign!Conv_layer_2/biases/Variable/Adamsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_8/tensor_namesConst*8
value/B-B#Conv_layer_2/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assign#Conv_layer_2/biases/Variable/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_2/biases/Variable*
_output_shapes
:(
�
save/RestoreV2_9/tensor_namesConst*2
value)B'BConv_layer_2/weights/Variable*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9AssignConv_layer_2/weights/Variablesave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_10/tensor_namesConst*7
value.B,B"Conv_layer_2/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign"Conv_layer_2/weights/Variable/Adamsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_11/tensor_namesConst*9
value0B.B$Conv_layer_2/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assign$Conv_layer_2/weights/Variable/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_2/weights/Variable*&
_output_shapes
:(
�
save/RestoreV2_12/tensor_namesConst*1
value(B&BConv_layer_3/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12AssignConv_layer_3/biases/Variablesave/RestoreV2_12*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_13/tensor_namesConst*6
value-B+B!Conv_layer_3/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assign!Conv_layer_3/biases/Variable/Adamsave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_14/tensor_namesConst*8
value/B-B#Conv_layer_3/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign#Conv_layer_3/biases/Variable/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_3/biases/Variable*
_output_shapes
:<
�
save/RestoreV2_15/tensor_namesConst*2
value)B'BConv_layer_3/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15AssignConv_layer_3/weights/Variablesave/RestoreV2_15*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_16/tensor_namesConst*7
value.B,B"Conv_layer_3/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign"Conv_layer_3/weights/Variable/Adamsave/RestoreV2_16*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_17/tensor_namesConst*9
value0B.B$Conv_layer_3/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign$Conv_layer_3/weights/Variable/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_3/weights/Variable*&
_output_shapes
:(<
�
save/RestoreV2_18/tensor_namesConst*1
value(B&BConv_layer_4/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18AssignConv_layer_4/biases/Variablesave/RestoreV2_18*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_19/tensor_namesConst*6
value-B+B!Conv_layer_4/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assign!Conv_layer_4/biases/Variable/Adamsave/RestoreV2_19*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_20/tensor_namesConst*8
value/B-B#Conv_layer_4/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign#Conv_layer_4/biases/Variable/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_4/biases/Variable*
_output_shapes
:P
�
save/RestoreV2_21/tensor_namesConst*2
value)B'BConv_layer_4/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_21AssignConv_layer_4/weights/Variablesave/RestoreV2_21*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
save/RestoreV2_22/tensor_namesConst*7
value.B,B"Conv_layer_4/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_22Assign"Conv_layer_4/weights/Variable/Adamsave/RestoreV2_22*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
�
save/RestoreV2_23/tensor_namesConst*9
value0B.B$Conv_layer_4/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_23Assign$Conv_layer_4/weights/Variable/Adam_1save/RestoreV2_23*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@Conv_layer_4/weights/Variable*&
_output_shapes
:<P
~
save/RestoreV2_24/tensor_namesConst*,
value#B!BDeepID1/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_24AssignDeepID1/biases/Variablesave/RestoreV2_24*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
save/RestoreV2_25/tensor_namesConst*1
value(B&BDeepID1/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_25AssignDeepID1/biases/Variable/Adamsave/RestoreV2_25*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�
�
save/RestoreV2_26/tensor_namesConst*3
value*B(BDeepID1/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_26AssignDeepID1/biases/Variable/Adam_1save/RestoreV2_26*
T0*
validate_shape(*
use_locking(**
_class 
loc:@DeepID1/biases/Variable*
_output_shapes	
:�

save/RestoreV2_27/tensor_namesConst*-
value$B"BDeepID1/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_27AssignDeepID1/weights/Variablesave/RestoreV2_27*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_28/tensor_namesConst*2
value)B'BDeepID1/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_28AssignDeepID1/weights/Variable/Adamsave/RestoreV2_28*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_29/tensor_namesConst*4
value+B)BDeepID1/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_29AssignDeepID1/weights/Variable/Adam_1save/RestoreV2_29*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@DeepID1/weights/Variable* 
_output_shapes
:
��
�
save/RestoreV2_30/tensor_namesConst*/
value&B$BDeepID1/weights_1/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30AssignDeepID1/weights_1/Variablesave/RestoreV2_30*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_31/tensor_namesConst*4
value+B)BDeepID1/weights_1/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_31AssignDeepID1/weights_1/Variable/Adamsave/RestoreV2_31*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_32/tensor_namesConst*6
value-B+B!DeepID1/weights_1/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assign!DeepID1/weights_1/Variable/Adam_1save/RestoreV2_32*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@DeepID1/weights_1/Variable* 
_output_shapes
:
�
�
�
save/RestoreV2_33/tensor_namesConst*2
value)B'Bloss/nn_layer/biases/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_33Assignloss/nn_layer/biases/Variablesave/RestoreV2_33*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_34/tensor_namesConst*7
value.B,B"loss/nn_layer/biases/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assign"loss/nn_layer/biases/Variable/Adamsave/RestoreV2_34*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_35/tensor_namesConst*9
value0B.B$loss/nn_layer/biases/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_35Assign$loss/nn_layer/biases/Variable/Adam_1save/RestoreV2_35*
T0*
validate_shape(*
use_locking(*0
_class&
$"loc:@loss/nn_layer/biases/Variable*
_output_shapes	
:�

�
save/RestoreV2_36/tensor_namesConst*3
value*B(Bloss/nn_layer/weights/Variable*
dtype0*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_36Assignloss/nn_layer/weights/Variablesave/RestoreV2_36*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
save/RestoreV2_37/tensor_namesConst*8
value/B-B#loss/nn_layer/weights/Variable/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_37Assign#loss/nn_layer/weights/Variable/Adamsave/RestoreV2_37*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

�
save/RestoreV2_38/tensor_namesConst*:
value1B/B%loss/nn_layer/weights/Variable/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_38Assign%loss/nn_layer/weights/Variable/Adam_1save/RestoreV2_38*
T0*
validate_shape(*
use_locking(*1
_class'
%#loc:@loss/nn_layer/weights/Variable* 
_output_shapes
:
��

x
save/RestoreV2_39/tensor_namesConst*&
valueBBtrain/beta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_39Assigntrain/beta1_powersave/RestoreV2_39*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
x
save/RestoreV2_40/tensor_namesConst*&
valueBBtrain/beta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_40Assigntrain/beta2_powersave/RestoreV2_40*
T0*
validate_shape(*
use_locking(*/
_class%
#!loc:@Conv_layer_1/biases/Variable*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40
�
initNoOp%^Conv_layer_1/weights/Variable/Assign$^Conv_layer_1/biases/Variable/Assign%^Conv_layer_2/weights/Variable/Assign$^Conv_layer_2/biases/Variable/Assign%^Conv_layer_3/weights/Variable/Assign$^Conv_layer_3/biases/Variable/Assign%^Conv_layer_4/weights/Variable/Assign$^Conv_layer_4/biases/Variable/Assign ^DeepID1/weights/Variable/Assign"^DeepID1/weights_1/Variable/Assign^DeepID1/biases/Variable/Assign&^loss/nn_layer/weights/Variable/Assign%^loss/nn_layer/biases/Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign*^Conv_layer_1/weights/Variable/Adam/Assign,^Conv_layer_1/weights/Variable/Adam_1/Assign)^Conv_layer_1/biases/Variable/Adam/Assign+^Conv_layer_1/biases/Variable/Adam_1/Assign*^Conv_layer_2/weights/Variable/Adam/Assign,^Conv_layer_2/weights/Variable/Adam_1/Assign)^Conv_layer_2/biases/Variable/Adam/Assign+^Conv_layer_2/biases/Variable/Adam_1/Assign*^Conv_layer_3/weights/Variable/Adam/Assign,^Conv_layer_3/weights/Variable/Adam_1/Assign)^Conv_layer_3/biases/Variable/Adam/Assign+^Conv_layer_3/biases/Variable/Adam_1/Assign*^Conv_layer_4/weights/Variable/Adam/Assign,^Conv_layer_4/weights/Variable/Adam_1/Assign)^Conv_layer_4/biases/Variable/Adam/Assign+^Conv_layer_4/biases/Variable/Adam_1/Assign%^DeepID1/weights/Variable/Adam/Assign'^DeepID1/weights/Variable/Adam_1/Assign'^DeepID1/weights_1/Variable/Adam/Assign)^DeepID1/weights_1/Variable/Adam_1/Assign$^DeepID1/biases/Variable/Adam/Assign&^DeepID1/biases/Variable/Adam_1/Assign+^loss/nn_layer/weights/Variable/Adam/Assign-^loss/nn_layer/weights/Variable/Adam_1/Assign*^loss/nn_layer/biases/Variable/Adam/Assign,^loss/nn_layer/biases/Variable/Adam_1/Assign""�5
	variables�5�5
�
Conv_layer_1/weights/Variable:0$Conv_layer_1/weights/Variable/Assign$Conv_layer_1/weights/Variable/read:02'Conv_layer_1/weights/truncated_normal:0
�
Conv_layer_1/biases/Variable:0#Conv_layer_1/biases/Variable/Assign#Conv_layer_1/biases/Variable/read:02Conv_layer_1/biases/zeros:0
�
Conv_layer_2/weights/Variable:0$Conv_layer_2/weights/Variable/Assign$Conv_layer_2/weights/Variable/read:02'Conv_layer_2/weights/truncated_normal:0
�
Conv_layer_2/biases/Variable:0#Conv_layer_2/biases/Variable/Assign#Conv_layer_2/biases/Variable/read:02Conv_layer_2/biases/zeros:0
�
Conv_layer_3/weights/Variable:0$Conv_layer_3/weights/Variable/Assign$Conv_layer_3/weights/Variable/read:02'Conv_layer_3/weights/truncated_normal:0
�
Conv_layer_3/biases/Variable:0#Conv_layer_3/biases/Variable/Assign#Conv_layer_3/biases/Variable/read:02Conv_layer_3/biases/zeros:0
�
Conv_layer_4/weights/Variable:0$Conv_layer_4/weights/Variable/Assign$Conv_layer_4/weights/Variable/read:02'Conv_layer_4/weights/truncated_normal:0
�
Conv_layer_4/biases/Variable:0#Conv_layer_4/biases/Variable/Assign#Conv_layer_4/biases/Variable/read:02Conv_layer_4/biases/zeros:0
�
DeepID1/weights/Variable:0DeepID1/weights/Variable/AssignDeepID1/weights/Variable/read:02"DeepID1/weights/truncated_normal:0
�
DeepID1/weights_1/Variable:0!DeepID1/weights_1/Variable/Assign!DeepID1/weights_1/Variable/read:02$DeepID1/weights_1/truncated_normal:0
s
DeepID1/biases/Variable:0DeepID1/biases/Variable/AssignDeepID1/biases/Variable/read:02DeepID1/biases/zeros:0
�
 loss/nn_layer/weights/Variable:0%loss/nn_layer/weights/Variable/Assign%loss/nn_layer/weights/Variable/read:02(loss/nn_layer/weights/truncated_normal:0
�
loss/nn_layer/biases/Variable:0$loss/nn_layer/biases/Variable/Assign$loss/nn_layer/biases/Variable/read:02loss/nn_layer/biases/zeros:0
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
$Conv_layer_1/weights/Variable/Adam:0)Conv_layer_1/weights/Variable/Adam/Assign)Conv_layer_1/weights/Variable/Adam/read:026Conv_layer_1/weights/Variable/Adam/Initializer/zeros:0
�
&Conv_layer_1/weights/Variable/Adam_1:0+Conv_layer_1/weights/Variable/Adam_1/Assign+Conv_layer_1/weights/Variable/Adam_1/read:028Conv_layer_1/weights/Variable/Adam_1/Initializer/zeros:0
�
#Conv_layer_1/biases/Variable/Adam:0(Conv_layer_1/biases/Variable/Adam/Assign(Conv_layer_1/biases/Variable/Adam/read:025Conv_layer_1/biases/Variable/Adam/Initializer/zeros:0
�
%Conv_layer_1/biases/Variable/Adam_1:0*Conv_layer_1/biases/Variable/Adam_1/Assign*Conv_layer_1/biases/Variable/Adam_1/read:027Conv_layer_1/biases/Variable/Adam_1/Initializer/zeros:0
�
$Conv_layer_2/weights/Variable/Adam:0)Conv_layer_2/weights/Variable/Adam/Assign)Conv_layer_2/weights/Variable/Adam/read:026Conv_layer_2/weights/Variable/Adam/Initializer/zeros:0
�
&Conv_layer_2/weights/Variable/Adam_1:0+Conv_layer_2/weights/Variable/Adam_1/Assign+Conv_layer_2/weights/Variable/Adam_1/read:028Conv_layer_2/weights/Variable/Adam_1/Initializer/zeros:0
�
#Conv_layer_2/biases/Variable/Adam:0(Conv_layer_2/biases/Variable/Adam/Assign(Conv_layer_2/biases/Variable/Adam/read:025Conv_layer_2/biases/Variable/Adam/Initializer/zeros:0
�
%Conv_layer_2/biases/Variable/Adam_1:0*Conv_layer_2/biases/Variable/Adam_1/Assign*Conv_layer_2/biases/Variable/Adam_1/read:027Conv_layer_2/biases/Variable/Adam_1/Initializer/zeros:0
�
$Conv_layer_3/weights/Variable/Adam:0)Conv_layer_3/weights/Variable/Adam/Assign)Conv_layer_3/weights/Variable/Adam/read:026Conv_layer_3/weights/Variable/Adam/Initializer/zeros:0
�
&Conv_layer_3/weights/Variable/Adam_1:0+Conv_layer_3/weights/Variable/Adam_1/Assign+Conv_layer_3/weights/Variable/Adam_1/read:028Conv_layer_3/weights/Variable/Adam_1/Initializer/zeros:0
�
#Conv_layer_3/biases/Variable/Adam:0(Conv_layer_3/biases/Variable/Adam/Assign(Conv_layer_3/biases/Variable/Adam/read:025Conv_layer_3/biases/Variable/Adam/Initializer/zeros:0
�
%Conv_layer_3/biases/Variable/Adam_1:0*Conv_layer_3/biases/Variable/Adam_1/Assign*Conv_layer_3/biases/Variable/Adam_1/read:027Conv_layer_3/biases/Variable/Adam_1/Initializer/zeros:0
�
$Conv_layer_4/weights/Variable/Adam:0)Conv_layer_4/weights/Variable/Adam/Assign)Conv_layer_4/weights/Variable/Adam/read:026Conv_layer_4/weights/Variable/Adam/Initializer/zeros:0
�
&Conv_layer_4/weights/Variable/Adam_1:0+Conv_layer_4/weights/Variable/Adam_1/Assign+Conv_layer_4/weights/Variable/Adam_1/read:028Conv_layer_4/weights/Variable/Adam_1/Initializer/zeros:0
�
#Conv_layer_4/biases/Variable/Adam:0(Conv_layer_4/biases/Variable/Adam/Assign(Conv_layer_4/biases/Variable/Adam/read:025Conv_layer_4/biases/Variable/Adam/Initializer/zeros:0
�
%Conv_layer_4/biases/Variable/Adam_1:0*Conv_layer_4/biases/Variable/Adam_1/Assign*Conv_layer_4/biases/Variable/Adam_1/read:027Conv_layer_4/biases/Variable/Adam_1/Initializer/zeros:0
�
DeepID1/weights/Variable/Adam:0$DeepID1/weights/Variable/Adam/Assign$DeepID1/weights/Variable/Adam/read:021DeepID1/weights/Variable/Adam/Initializer/zeros:0
�
!DeepID1/weights/Variable/Adam_1:0&DeepID1/weights/Variable/Adam_1/Assign&DeepID1/weights/Variable/Adam_1/read:023DeepID1/weights/Variable/Adam_1/Initializer/zeros:0
�
!DeepID1/weights_1/Variable/Adam:0&DeepID1/weights_1/Variable/Adam/Assign&DeepID1/weights_1/Variable/Adam/read:023DeepID1/weights_1/Variable/Adam/Initializer/zeros:0
�
#DeepID1/weights_1/Variable/Adam_1:0(DeepID1/weights_1/Variable/Adam_1/Assign(DeepID1/weights_1/Variable/Adam_1/read:025DeepID1/weights_1/Variable/Adam_1/Initializer/zeros:0
�
DeepID1/biases/Variable/Adam:0#DeepID1/biases/Variable/Adam/Assign#DeepID1/biases/Variable/Adam/read:020DeepID1/biases/Variable/Adam/Initializer/zeros:0
�
 DeepID1/biases/Variable/Adam_1:0%DeepID1/biases/Variable/Adam_1/Assign%DeepID1/biases/Variable/Adam_1/read:022DeepID1/biases/Variable/Adam_1/Initializer/zeros:0
�
%loss/nn_layer/weights/Variable/Adam:0*loss/nn_layer/weights/Variable/Adam/Assign*loss/nn_layer/weights/Variable/Adam/read:027loss/nn_layer/weights/Variable/Adam/Initializer/zeros:0
�
'loss/nn_layer/weights/Variable/Adam_1:0,loss/nn_layer/weights/Variable/Adam_1/Assign,loss/nn_layer/weights/Variable/Adam_1/read:029loss/nn_layer/weights/Variable/Adam_1/Initializer/zeros:0
�
$loss/nn_layer/biases/Variable/Adam:0)loss/nn_layer/biases/Variable/Adam/Assign)loss/nn_layer/biases/Variable/Adam/read:026loss/nn_layer/biases/Variable/Adam/Initializer/zeros:0
�
&loss/nn_layer/biases/Variable/Adam_1:0+loss/nn_layer/biases/Variable/Adam_1/Assign+loss/nn_layer/biases/Variable/Adam_1/read:028loss/nn_layer/biases/Variable/Adam_1/Initializer/zeros:0"�
trainable_variables��
�
Conv_layer_1/weights/Variable:0$Conv_layer_1/weights/Variable/Assign$Conv_layer_1/weights/Variable/read:02'Conv_layer_1/weights/truncated_normal:0
�
Conv_layer_1/biases/Variable:0#Conv_layer_1/biases/Variable/Assign#Conv_layer_1/biases/Variable/read:02Conv_layer_1/biases/zeros:0
�
Conv_layer_2/weights/Variable:0$Conv_layer_2/weights/Variable/Assign$Conv_layer_2/weights/Variable/read:02'Conv_layer_2/weights/truncated_normal:0
�
Conv_layer_2/biases/Variable:0#Conv_layer_2/biases/Variable/Assign#Conv_layer_2/biases/Variable/read:02Conv_layer_2/biases/zeros:0
�
Conv_layer_3/weights/Variable:0$Conv_layer_3/weights/Variable/Assign$Conv_layer_3/weights/Variable/read:02'Conv_layer_3/weights/truncated_normal:0
�
Conv_layer_3/biases/Variable:0#Conv_layer_3/biases/Variable/Assign#Conv_layer_3/biases/Variable/read:02Conv_layer_3/biases/zeros:0
�
Conv_layer_4/weights/Variable:0$Conv_layer_4/weights/Variable/Assign$Conv_layer_4/weights/Variable/read:02'Conv_layer_4/weights/truncated_normal:0
�
Conv_layer_4/biases/Variable:0#Conv_layer_4/biases/Variable/Assign#Conv_layer_4/biases/Variable/read:02Conv_layer_4/biases/zeros:0
�
DeepID1/weights/Variable:0DeepID1/weights/Variable/AssignDeepID1/weights/Variable/read:02"DeepID1/weights/truncated_normal:0
�
DeepID1/weights_1/Variable:0!DeepID1/weights_1/Variable/Assign!DeepID1/weights_1/Variable/read:02$DeepID1/weights_1/truncated_normal:0
s
DeepID1/biases/Variable:0DeepID1/biases/Variable/AssignDeepID1/biases/Variable/read:02DeepID1/biases/zeros:0
�
 loss/nn_layer/weights/Variable:0%loss/nn_layer/weights/Variable/Assign%loss/nn_layer/weights/Variable/read:02(loss/nn_layer/weights/truncated_normal:0
�
loss/nn_layer/biases/Variable:0$loss/nn_layer/biases/Variable/Assign$loss/nn_layer/biases/Variable/read:02loss/nn_layer/biases/zeros:0"3
	summaries&
$
loss/loss:0
accuracy/accuracy_1:0"
train_op


train/Adam9A@y9       �7�	R �����A*.

	loss/loss�	bD

accuracy/accuracy_1�t�9�8s�;       #�\	Ob^����Ad*.

	loss/loss�<�@

accuracy/accuracy_15RL:E��<       ȷ�R	̪�����A�*.

	loss/loss�:�@

accuracy/accuracy_15RL:#�߳<       ȷ�R	�+�û��A�*.

	loss/loss�d�@

accuracy/accuracy_15RL:
�?<       ȷ�R	��ƻ��A�*.

	loss/loss�@

accuracy/accuracy_15RL:fh<       ȷ�R	vkȻ��A�*.

	loss/lossD��@

accuracy/accuracy_1s/u:�қL<       ȷ�R	���ʻ��A�*.

	loss/loss��@

accuracy/accuracy_1s/u:��?)<       ȷ�R	�Lͻ��A�*.

	loss/loss��@

accuracy/accuracy_1#��:��D�<       ȷ�R	5Bϻ��A�*.

	loss/loss``�@

accuracy/accuracy_1#��:z�=<       ȷ�R	�?~ѻ��A�*.

	loss/loss^:�@

accuracy/accuracy_1�f�:�n�F<       ȷ�R	Q�ӻ��A�*.

	loss/loss��@

accuracy/accuracy_1��7;���?<       ȷ�R	�U\ֻ��A�*.

	loss/loss*��@

accuracy/accuracy_1Kz;��<       ȷ�R	��lػ��A�	*.

	loss/lossߙ�@

accuracy/accuracy_1��;~f`�<       ȷ�R	�ťڻ��A�
*.

	loss/loss�W�@

accuracy/accuracy_1ԯ�;xb@+<       ȷ�R	a&ݻ��A�
*.

	loss/loss
'�@

accuracy/accuracy_1:�;&��i<       ȷ�R	b)�߻��A�*.

	loss/loss���@

accuracy/accuracy_1>��;�"Hb<       ȷ�R	 �Ồ�A�*.

	loss/lossē�@

accuracy/accuracy_1�U�;�?�<       ȷ�R	���㻒�A�*.

	loss/loss�L�@

accuracy/accuracy_1��;��><       ȷ�R	@�>滒�A�*.

	loss/loss���@

accuracy/accuracy_1�m�;!�/�<       ȷ�R	�ū軒�A�*.

	loss/loss���@

accuracy/accuracy_1|��;ߕ�~<       ȷ�R	J-뻒�A�*.

	loss/lossA�@

accuracy/accuracy_1���;�A�<       ȷ�R	�&*����A�*.

	loss/loss��@

accuracy/accuracy_1�x<�׿M<       ȷ�R	��`ﻒ�A�*.

	loss/loss��@

accuracy/accuracy_1�4<z~��<       ȷ�R	7���A�*.

	loss/loss)_�@

accuracy/accuracy_1>�<<�S��<       ȷ�R	��9����A�*.

	loss/lossg��@

accuracy/accuracy_1n�W<~>��<       ȷ�R	p������A�*.

	loss/loss?��@

accuracy/accuracy_1�Zq<ʷz<       ȷ�R	������A�*.

	loss/lossA��@

accuracy/accuracy_1��r< �O<       ȷ�R	������A�*.

	loss/loss���@

accuracy/accuracy_1R �<���<       ȷ�R	��b����A�*.

	loss/loss�$�@

accuracy/accuracy_1�n�<-�|<       ȷ�R	W)�����A�*.

	loss/loss��@

accuracy/accuracy_1G��<X��V<       ȷ�R	 {����A�*.

	loss/loss���@

accuracy/accuracy_1���<����<       ȷ�R	�L���A�*.

	loss/loss��@

accuracy/accuracy_1��<l<       ȷ�R	k����A�*.

	loss/losss��@

accuracy/accuracy_1���<�E<       ȷ�R	W�����A�*.

	loss/loss |�@

accuracy/accuracy_1Ǵ�<|~0�<       ȷ�R	��j���A�*.

	loss/loss���@

accuracy/accuracy_1�N�<(�EI<       ȷ�R	�2_���A�*.

	loss/loss��@

accuracy/accuracy_1f#�<*,�C<       ȷ�R	�Q����A�*.

	loss/lossV*�@

accuracy/accuracy_1��<FM��<       ȷ�R	�r;���A�*.

	loss/loss/�@

accuracy/accuracy_1~r=m��<       ȷ�R	�W����A�*.

	loss/lossC�@

accuracy/accuracy_10=}ڌ<       ȷ�R	�z����A�*.

	loss/loss���@

accuracy/accuracy_1f=���<       ȷ�R	�����A�*.

	loss/loss���@

accuracy/accuracy_1�>(=j��<       ȷ�R	5(^���A� *.

	loss/loss���@

accuracy/accuracy_1��7=W��<       ȷ�R	������A� *.

	loss/lossD��@

accuracy/accuracy_1m�==��O><       ȷ�R	z�+ ���A�!*.

	loss/loss�C�@

accuracy/accuracy_1&�T=f���<       ȷ�R	]�"���A�"*.

	loss/losso"�@

accuracy/accuracy_1b�c=h٨�<       ȷ�R	!��$���A�#*.

	loss/loss5�@

accuracy/accuracy_1�ep=uMF<       ȷ�R	%/�&���A�#*.

	loss/loss���@

accuracy/accuracy_1z5|=!�<       ȷ�R	Idm)���A�$*.

	loss/loss��@

accuracy/accuracy_1�g�=zxi?<       ȷ�R	���+���A�%*.

	loss/loss���@

accuracy/accuracy_1���=����<       ȷ�R	zW�-���A�&*.

	loss/loss��@

accuracy/accuracy_19��=%T�8<       ȷ�R	��&0���A�'*.

	loss/loss`\�@

accuracy/accuracy_1��=��=<       ȷ�R	�.�2���A�'*.

	loss/lossL��@

accuracy/accuracy_1L�=#M�t<       ȷ�R	�N5���A�(*.

	loss/loss ��@

accuracy/accuracy_1*��=Q�ŵ<       ȷ�R	Mw7���A�)*.

	loss/loss�9�@

accuracy/accuracy_1�=���o<       ȷ�R	?�c9���A�**.

	loss/loss���@

accuracy/accuracy_1>��=b��<       ȷ�R	jH�;���A�**.

	loss/loss6�@

accuracy/accuracy_1�a�=,�[<       ȷ�R	w*>���A�+*.

	loss/loss*ܾ@

accuracy/accuracy_1 E�=����<       ȷ�R	�5D@���A�,*.

	loss/loss�=�@

accuracy/accuracy_1��=>��U<       ȷ�R	#kB���A�-*.

	loss/loss���@

accuracy/accuracy_1��=e��<       ȷ�R	��D���A�.*.

	loss/loss䬺@

accuracy/accuracy_1|��= 
�V<       ȷ�R	\V0G���A�.*.

	loss/lossNj�@

accuracy/accuracy_1"� >���<       ȷ�R	�n�I���A�/*.

	loss/loss1�@

accuracy/accuracy_1N}>I<       ȷ�R	�vK���A�0*.

	loss/loss-�@

accuracy/accuracy_1��>�9�<       ȷ�R	d��M���A�1*.

	loss/loss\س@

accuracy/accuracy_13#>��<       ȷ�R	�JAP���A�2*.

	loss/loss�+�@

accuracy/accuracy_1)�>��<       ȷ�R	x��R���A�2*.

	loss/lossn��@

accuracy/accuracy_1PY>?�<       ȷ�R	���T���A�3*.

	loss/loss N�@

accuracy/accuracy_1� >���<       ȷ�R	`g�V���A�4*.

	loss/loss���@

accuracy/accuracy_1@�&>8G�a<       ȷ�R	�tdY���A�5*.

	loss/lossC��@

accuracy/accuracy_1J�*>�i�<       ȷ�R	�j�[���A�5*.

	loss/loss$�@

accuracy/accuracy_1��/>����<       ȷ�R	�+^���A�6*.

	loss/loss�/�@

accuracy/accuracy_1��7>�9��<       ȷ�R	2=`���A�7*.

	loss/loss��@

accuracy/accuracy_1(F>>�L�T<       ȷ�R	ַzb���A�8*.

	loss/lossQ��@

accuracy/accuracy_1>vC>I��<       ȷ�R	1��d���A�9*.

	loss/lossM��@

accuracy/accuracy_1^�L>R�<       ȷ�R	��Hg���A�9*.

	loss/loss١@

accuracy/accuracy_1��P>�OC<       ȷ�R	�ji���A�:*.

	loss/loss*b�@

accuracy/accuracy_1��Q>ш��<       ȷ�R	$W�k���A�;*.

	loss/loss�(�@

accuracy/accuracy_1x�[>�b�<       ȷ�R	���m���A�<*.

	loss/loss��@

accuracy/accuracy_1o�d>��<       ȷ�R	�\p���A�<*.

	loss/loss���@

accuracy/accuracy_1?l>��;#<       ȷ�R	\�r���A�=*.

	loss/lossކ�@

accuracy/accuracy_1�Zq>��@-<       ȷ�R	A�t���A�>*.

	loss/losskM�@

accuracy/accuracy_1p#x>g�ɿ<       ȷ�R	L�w���A�?*.

	loss/loss�ʕ@

accuracy/accuracy_1}Ay>FR�t<       ȷ�R	WHpy���A�@*.

	loss/loss�&�@

accuracy/accuracy_1[��>y6��<       ȷ�R	�p�{���A�@*.

	loss/loss�m�@

accuracy/accuracy_1���>���<       ȷ�R	���}���A�A*.

	loss/lossW��@

accuracy/accuracy_1�1�>�@<       ȷ�R	'� ����A�B*.

	loss/loss�I�@

accuracy/accuracy_1�#�>�ئ�<       ȷ�R	4b�����A�C*.

	loss/loss@

accuracy/accuracy_1���>��7<       ȷ�R	�ꄼ��A�C*.

	loss/loss���@

accuracy/accuracy_1g��>ܸ!<       ȷ�R	�J����A�D*.

	loss/loss��@

accuracy/accuracy_1Nӟ>A;��<       ȷ�R	��5����A�E*.

	loss/loss.��@

accuracy/accuracy_1��>㙉<       ȷ�R	K������A�F*.

	loss/loss$��@

accuracy/accuracy_1"ب>�Y�<       ȷ�R	?J�����A�G*.

	loss/lossY{@

accuracy/accuracy_1��>��j�<       ȷ�R	� ]����A�G*.

	loss/loss��w@

accuracy/accuracy_1j��>uQ<       ȷ�R	�큒���A�H*.

	loss/lossU�s@

accuracy/accuracy_1��>$l�<       ȷ�R	މ�����A�I*.

	loss/lossE�q@

accuracy/accuracy_1 f�>:��$<       ȷ�R	W�
����A�J*.

	loss/losslak@

accuracy/accuracy_16a�>����<       ȷ�R	M~n����A�K*.

	loss/loss�Md@

accuracy/accuracy_1K�>���<       ȷ�R	I}ϛ���A�K*.

	loss/loss�b@

accuracy/accuracy_1k��>QU�p<       ȷ�R	����A�L*.

	loss/loss��[@

accuracy/accuracy_1	W�>�j�Z<       ȷ�R	O(����A�M*.

	loss/loss`�X@

accuracy/accuracy_1���>���<       ȷ�R	������A�N*.

	loss/lossM�S@

accuracy/accuracy_1Ǣ�>�=��<       ȷ�R	��2����A�N*.

	loss/loss��P@

accuracy/accuracy_1���>�R��<       ȷ�R	�X����A�O*.

	loss/lossTSM@

accuracy/accuracy_1�L�>/>�<       ȷ�R	�t����A�P*.

	loss/lossZ�F@

accuracy/accuracy_10�>��<       ȷ�R	kF۫���A�Q*.

	loss/loss�hB@

accuracy/accuracy_1=��>�8M<       ȷ�R	�?����A�R*.

	loss/loss��=@

accuracy/accuracy_1�_�>���X<       ȷ�R	�x�����A�R*.

	loss/loss�7@

accuracy/accuracy_1�# ?wg�<       ȷ�R	f�����A�S*.

	loss/loss�5@

accuracy/accuracy_1��?�s�G<       ȷ�R	\&촼��A�T*.

	loss/loss�w1@

accuracy/accuracy_1��?��y0<       ȷ�R	$�R����A�U*.

	loss/lossֺ-@

accuracy/accuracy_1T�?� ��<       ȷ�R	:�����A�U*.

	loss/lossI�(@

accuracy/accuracy_1R ?at�<       ȷ�R	�Ỽ��A�V*.

	loss/loss�#@

accuracy/accuracy_1[?�g�<       ȷ�R	�:�����A�W*.

	loss/loss�@

accuracy/accuracy_1��?�{�s<       ȷ�R	�a����A�X*.

	loss/lossv@

accuracy/accuracy_1��?~}h&<       ȷ�R	}��¼��A�Y*.

	loss/loss�V@

accuracy/accuracy_1�I?��<       ȷ�R	�52ż��A�Y*.

	loss/loss\@

accuracy/accuracy_1G�?��_~<       ȷ�R	�LǼ��A�Z*.

	loss/loss�(@

accuracy/accuracy_1)�?�9.<       ȷ�R	}'rɼ��A�[*.

	loss/loss��@

accuracy/accuracy_1�?m��=<       ȷ�R	�Q�˼��A�\*.

	loss/loss�H@

accuracy/accuracy_1�#"?;���<       ȷ�R	N�6μ��A�\*.

	loss/loss�2�?

accuracy/accuracy_1��$?M���<       ȷ�R	�iaм��A�]*.

	loss/loss~X�?

accuracy/accuracy_1��&?���<       ȷ�R	�{Ҽ��A�^*.

	loss/loss���?

accuracy/accuracy_1�8*?�`��<       ȷ�R	Gg�Լ��A�_*.

	loss/loss�i�?

accuracy/accuracy_1
V,?~��0<       ȷ�R	��A׼��A�`*.

	loss/loss$s�?

accuracy/accuracy_1!�0?���^<       ȷ�R	�ټ��A�`*.

	loss/loss�J�?

accuracy/accuracy_1]�2?CVD<       ȷ�R	c9�ۼ��A�a*.

	loss/loss��?

accuracy/accuracy_1p3?V���<       ȷ�R	j$�ݼ��A�b*.

	loss/loss�'�?

accuracy/accuracy_1��7?K�Ў<       ȷ�R	��K༒�A�c*.

	loss/loss���?

accuracy/accuracy_1�9?? <       ȷ�R	)��⼒�A�d*.

	loss/lossTz�?

accuracy/accuracy_1�:?b��<       ȷ�R	Pt�伒�A�d*.

	loss/loss��?

accuracy/accuracy_1V=?0GC)<       ȷ�R	Y��漒�A�e*.

	loss/loss�.�?

accuracy/accuracy_1�T??*�q�<       ȷ�R	�	U鼒�A�f*.

	loss/loss��?

accuracy/accuracy_1u=D?���<       ȷ�R	���뼒�A�g*.

	loss/loss��?

accuracy/accuracy_1P�F?+�c�<       ȷ�R	�� �A�g*.

	loss/loss��?

accuracy/accuracy_1�JG?�ua'<       ȷ�R	����A�h*.

	loss/loss��?

accuracy/accuracy_1Y�J?
��<<       ȷ�R	�3i��A�i*.

	loss/losscm�?

accuracy/accuracy_1�$K?ۨ�-<       ȷ�R	J=�����A�j*.

	loss/lossSW�?

accuracy/accuracy_1��K?�]X<       ȷ�R	��7����A�k*.

	loss/lossrɌ?

accuracy/accuracy_1$�O?A�&�<       ȷ�R	;ki����A�k*.

	loss/loss@E�?

accuracy/accuracy_1@�N?�j<       ȷ�R	�;�����A�l*.

	loss/loss�O�?

accuracy/accuracy_1D�Q?�&�<       ȷ�R	v	�����A�m*.

	loss/loss~?

accuracy/accuracy_1�S?,�@&<       ȷ�R	X�R ���A�n*.

	loss/loss��|?

accuracy/accuracy_1��S?���(<       ȷ�R	����A�n*.

	loss/loss�p?

accuracy/accuracy_1GV?��<       ȷ�R	�S����A�o*.

	loss/loss�f?

accuracy/accuracy_10�W?�sy<       ȷ�R	f����A�p*.

	loss/lossw[e?

accuracy/accuracy_1R�X?Ћ�~<       ȷ�R	��l	���A�q*.

	loss/lossu�[?

accuracy/accuracy_1��Z?*��<       ȷ�R	ٔ����A�r*.

	loss/loss��V?

accuracy/accuracy_1r[?�PUX<       ȷ�R	�����A�r*.

	loss/loss"�S?

accuracy/accuracy_1�]?E��<       ȷ�R	����A�s*.

	loss/loss�Q?

accuracy/accuracy_12�\?�p�t<       ȷ�R	%����A�t*.

	loss/loss��`?

accuracy/accuracy_1��[?�5�<       ȷ�R	������A�u*.

	loss/loss�U?

accuracy/accuracy_1$M]?�WJ<       ȷ�R	U����A�u*.

	loss/loss�M?

accuracy/accuracy_1��_?ͱח<       ȷ�R	[ˀ���A�v*.

	loss/loss^AE?

accuracy/accuracy_1�_?�
�9<       ȷ�R	(����A�w*.

	loss/loss�=?

accuracy/accuracy_1,a?�`>�<       ȷ�R	�.���A�x*.

	loss/loss�b??

accuracy/accuracy_1�@a?w��d<       ȷ�R	��� ���A�y*.

	loss/loss<X8?

accuracy/accuracy_1�a?�v;p<       ȷ�R	���"���A�y*.

	loss/loss��7?

accuracy/accuracy_1��b?P�J<       ȷ�R	e��$���A�z*.

	loss/loss9�<?

accuracy/accuracy_1sb?��W�<       ȷ�R	�7A'���A�{*.

	loss/loss��*?

accuracy/accuracy_17qe?%6��<       ȷ�R	���)���A�|*.

	loss/loss�*?

accuracy/accuracy_1��d?v��<       ȷ�R	��,���A�}*.

	loss/loss�?(?

accuracy/accuracy_1ֽe?�C<       ȷ�R	}.���A�}*.

	loss/lossJ�>?

accuracy/accuracy_1sb?a�n�<       ȷ�R	T0���A�~*.

	loss/lossK�)?

accuracy/accuracy_1h�d?���<       ȷ�R	���2���A�*.

	loss/loss�u?

accuracy/accuracy_1CLg?��=       `I��	�� 5���A��*.

	loss/lossr!?

accuracy/accuracy_1�jg?hj-�=       `I��	4�X7���A�*.

	loss/loss/� ?

accuracy/accuracy_1e�g?�+�=       `I��	��e9���A؁*.

	loss/loss��?

accuracy/accuracy_1&�h?�$x=       `I��	���;���A��*.

	loss/lossF�?

accuracy/accuracy_1�h?�=       `I��	j�5>���A��*.

	loss/loss@D#?

accuracy/accuracy_1��f?��"�=       `I��	�F�@���A��*.

	loss/lossݚ?

accuracy/accuracy_1��h?��0�=       `I��	���B���A�*.

	loss/loss%&?

accuracy/accuracy_1�Tj?�T�J=       `I��	��D���A̅*.

	loss/loss��?

accuracy/accuracy_1rAh?a'u=       `I��	��JG���A��*.

	loss/loss�w?

accuracy/accuracy_1��i?��(2=       `I��	揰I���A��*.

	loss/lossh??

accuracy/accuracy_1�Ik?_J�?=       `I��	���K���A��*.

	loss/loss�?

accuracy/accuracy_19nj?���=       `I��	��M���A܈*.

	loss/lossI�?

accuracy/accuracy_1ߌj?|E�T=       `I��	=�^P���A��*.

	loss/loss<�"?

accuracy/accuracy_1��g?3�yo=       `I��	�ſR���A��*.

	loss/lossz:?

accuracy/accuracy_1l?�l>�=       `I��	��#U���A��*.

	loss/loss��?

accuracy/accuracy_1�l?TΕ=       `I��	��%W���A�*.

	loss/loss�T?

accuracy/accuracy_1X�m?
R��=       `I��	W~hY���AЌ*.

	loss/loss�?

accuracy/accuracy_1`m?GVշ=       `I��	�[���A��*.

	loss/loss��?

accuracy/accuracy_1��k?0ax�=       `I��	�b-^���A��*.

	loss/loss�?

accuracy/accuracy_1dj?�LAK=       `I��	c�j`���A��*.

	loss/loss�f?

accuracy/accuracy_1��l?J^�/=       `I��	�Jqb���A��*.

	loss/loss9p?

accuracy/accuracy_1�m?���=       `I��	'��d���AĐ*.

	loss/loss��?

accuracy/accuracy_1f�m?���F=       `I��	�';g���A��*.

	loss/loss�O�>

accuracy/accuracy_1q2p?�C=       `I��	���i���A��*.

	loss/loss|�?

accuracy/accuracy_1<�m?���=       `I��	T�k���A�*.

	loss/loss_?

accuracy/accuracy_1B=o?F� =       `I��	���m���Aԓ*.

	loss/loss��?

accuracy/accuracy_1o?�N�h=       `I��	��Np���A��*.

	loss/lossb��>

accuracy/accuracy_1,�p?�� =       `I��	DǶr���A��*.

	loss/loss�y�>

accuracy/accuracy_1U�p?M�I�=       `I��	P�t���A��*.

	loss/loss��>

accuracy/accuracy_1��p?�a��=       `I��	��v���A�*.

	loss/loss[��>

accuracy/accuracy_1��p?���3=       `I��	�vey���Aȗ*.

	loss/loss��?

accuracy/accuracy_1ruo?���==       `I��	ٲ�{���A��*.

	loss/loss);�>

accuracy/accuracy_11�q?G6�p=       `I��	�4~���A��*.

	loss/lossh��>

accuracy/accuracy_1�Uq?{KIW=       `I��	m�<����A��*.

	loss/lossg��>

accuracy/accuracy_1�_q?�7�F=       `I��	�Mz����Aؚ*.

	loss/lossws�>

accuracy/accuracy_1�q?���=       `I��	��ބ���A��*.

	loss/loss��>

accuracy/accuracy_1�q??��=       `I��	8�D����A��*.

	loss/loss(�>

accuracy/accuracy_1��q?�=       `I��	2�щ���A��*.

	loss/lossh4�>

accuracy/accuracy_1�nr?�e�=       `I��	u�ҋ���A�*.

	loss/loss���>

accuracy/accuracy_1�q?1�G�=       `I��	"^4����A̞*.

	loss/loss�t�>

accuracy/accuracy_1?1r?"��=       `I��	� �����A��*.

	loss/loss�N�>

accuracy/accuracy_1p�q?"�"�=       `I��	�)�����A��*.

	loss/loss�=�>

accuracy/accuracy_1�q?߼�,=       `I��	�D����A��*.

	loss/loss|
�>

accuracy/accuracy_1�p?f��=       `I��	��?����Aܡ*.

	loss/loss�@�>

accuracy/accuracy_1�r?23=       `I��	K.�����A��*.

	loss/loss�Q�>

accuracy/accuracy_1�0s?r��B=       `I��	������A��*.

	loss/lossu0�>

accuracy/accuracy_1�0s?���m=       `I��	�F����A��*.

	loss/loss��>

accuracy/accuracy_1h�r?a+�=       `I��	��D����A�*.

	loss/loss���>

accuracy/accuracy_18_r?�I�=       `I��	�p�����AХ*.

	loss/loss���>

accuracy/accuracy_1Sdr?FAw=       `I��	Xt����A��*.

	loss/loss��>

accuracy/accuracy_1�5s?�]X=       `I��	gk�����A��*.

	loss/loss~��>

accuracy/accuracy_1��r?�� g=       `I��	5j�����A��*.

	loss/loss�F?

accuracy/accuracy_1�q?�$7�=       `I��	&̫���A�*.

	loss/lossEI�>

accuracy/accuracy_11�q?�1S!=       `I��	
�4����Aĩ*.

	loss/loss���>

accuracy/accuracy_1KOs?K�u�=       `I��	�������A��*.

	loss/loss ��>

accuracy/accuracy_1!�s?q�=       `I��	�粽��A��*.

	loss/loss���>

accuracy/accuracy_1!�s?�i��=       `I��	S4崽��A�*.

	loss/loss���>

accuracy/accuracy_1<ht?U�g=       `I��	/�J����AԬ*.

	loss/loss���>

accuracy/accuracy_1�*t?*�.9=       `I��	(>�����A��*.

	loss/loss��>

accuracy/accuracy_1{Dt?dvfX=       `I��	������A��*.

	loss/loss�0�>

accuracy/accuracy_15�t?���U=       `I��	YM*����A��*.

	loss/loss}C�>

accuracy/accuracy_1��t?����=       `I��	��`����A�*.

	loss/loss6��>

accuracy/accuracy_1��t?W��e=       `I��	��½��AȰ*.

	loss/lossX-�>

accuracy/accuracy_1^�t?ak&�=       `I��	4�)Ž��A��*.

	loss/loss���>

accuracy/accuracy_1�r?A�xh=       `I��	F�uǽ��A��*.

	loss/loss���>

accuracy/accuracy_1m�s?����=       `I��	x+qɽ��A��*.

	loss/lossu�?

accuracy/accuracy_1+Aq?,�1=       `I��	��˽��Aس*.

	loss/loss��>

accuracy/accuracy_1=�s?�Ba=       `I��	��:ν��A��*.

	loss/loss�h�>

accuracy/accuracy_1۴t?��/�=       `I��	Fs�н��A��*.

	loss/loss�>

accuracy/accuracy_1zu?�k�=       `I��	��ҽ��A��*.

	loss/lossw��>

accuracy/accuracy_1�Hu?�.�U=       `I��	A��Խ��A�*.

	loss/lossg��>

accuracy/accuracy_1�vu?�8�Z=       `I��	�sZ׽��A̷*.

	loss/loss��>

accuracy/accuracy_1��u?�`L=       `I��	v��ٽ��A��*.

	loss/loss��>

accuracy/accuracy_1;�u?�˅�=       `I��	�Pܽ��A��*.

	loss/lossM1�>

accuracy/accuracy_1�u?��=       `I��	0�޽��A��*.

	loss/loss\�>

accuracy/accuracy_1��u?�=       `I��	�qkདྷ�Aܺ*.

	loss/loss���>

accuracy/accuracy_1�gu?�E�S=       `I��	���⽒�A��*.

	loss/loss���>

accuracy/accuracy_1r�u?�)}�=       `I��	w�4归�A��*.

	loss/loss��>

accuracy/accuracy_14v?*M�=       `I��	#�J罒�A��*.

	loss/loss��>

accuracy/accuracy_1�u?}�=       `I��	:�z齒�A�*.

	loss/loss�k�>

accuracy/accuracy_1�u?��Y5=       `I��	ӗ�뽒�Aо*.

	loss/lossK�>

accuracy/accuracy_1��u?���q=       `I��	�3H�A��*.

	loss/loss���>

accuracy/accuracy_1y�u?ݍ�j=       `I��		���A��*.

	loss/lossh��>

accuracy/accuracy_1��u?��=       `I��	C����A��*.

	loss/loss��>

accuracy/accuracy_1�vu?��
=       `I��	������A��*.

	loss/loss���>

accuracy/accuracy_1�qu?��З=       `I��	��X����A��*.

	loss/lossl\�>

accuracy/accuracy_1Ov?�͚�=       `I��	�������A��*.

	loss/lossF5�>

accuracy/accuracy_1��u?�ZQa=       `I��	�l����A��*.

	loss/loss	��>

accuracy/accuracy_1��u?��X=       `I��	HB����A��*.

	loss/lossfD�>

accuracy/accuracy_1�v?\p�=       `I��	�� ���A��*.

	loss/loss���>

accuracy/accuracy_1�u?w�-�=       `I��	ܼ���A��*.

	loss/lossp�>

accuracy/accuracy_14Su?��=       `I��	��e���A��*.

	loss/lossD��>

accuracy/accuracy_1��u?O;��=       `I��	W���A��*.

	loss/loss&~�>

accuracy/accuracy_1�t?�p8�=       `I��	�J�	���A��*.

	loss/lossس?

accuracy/accuracy_1ûq?���=       `I��	�����A��*.

	loss/loss���>

accuracy/accuracy_1��u?�S�=       `I��	
����A��*.

	loss/loss���>

accuracy/accuracy_1�qu?pǀ=       `I��	�H����A��*.

	loss/loss���>

accuracy/accuracy_1�$v?u
M�=       `I��	�����A��*.

	loss/loss���>

accuracy/accuracy_1�v?$'��=       `I��	�n0���A��*.

	loss/lossڂ�>

accuracy/accuracy_1�v?�R�=       `I��	������A��*.

	loss/loss���>

accuracy/accuracy_1ĸv?�p��=       `I��	�����A��*.

	loss/loss��>

accuracy/accuracy_1��v?��uc=       `I��	#�����A��*.

	loss/loss3j�>

accuracy/accuracy_1�v?�Uc�=       `I��	k1=���A��*.

	loss/loss��>

accuracy/accuracy_1q�v?\���=       `I��	��� ���A��*.

	loss/lossp��>

accuracy/accuracy_1�v?��'-=       `I��	��#���A��*.

	loss/loss��>

accuracy/accuracy_1��v?͊�6=       `I��	�5%���A��*.

	loss/loss��>

accuracy/accuracy_1��v?��4�=       `I��	��`'���A��*.

	loss/loss���>

accuracy/accuracy_1O�v?k&D�=       `I��	���)���A��*.

	loss/losseD�>

accuracy/accuracy_1x{v?>6�&=       `I��	S�),���A��*.

	loss/loss��>

accuracy/accuracy_1�.v?��F�=       `I��	�.���A��*.

	loss/lossAy�>

accuracy/accuracy_1�av?�cX=       `I��	'�o0���A��*.

	loss/loss���>

accuracy/accuracy_1�v?�#X=       `I��	�s�2���A��*.

	loss/lossK��>

accuracy/accuracy_1O�v?Cu0|=       `I��	��65���A��*.

	loss/loss �>

accuracy/accuracy_1ˊv?}��=       `I��	(�7���A��*.

	loss/loss�v�>

accuracy/accuracy_1q�v?����=       `I��	���9���A��*.

	loss/loss$��>

accuracy/accuracy_1ss?��=       `I��	4��;���A��*.

	loss/loss���>

accuracy/accuracy_1��s?=Az�=       `I��	�O>���A��*.

	loss/lossa�>

accuracy/accuracy_1'�t?F�7=       `I��	���@���A��*.

	loss/loss��>

accuracy/accuracy_1<%u? �j=       `I��	�C���A��*.

	loss/loss�@�>

accuracy/accuracy_1�\v?)�H�=       `I��	� E���A��*.

	loss/lossZ��>

accuracy/accuracy_1ˊv?%��=       `I��	��aG���A��*.

	loss/loss���>

accuracy/accuracy_1Rw?��^T=       `I��	e�I���A��*.

	loss/lossu��>

accuracy/accuracy_1qfw?%=       `I��	�1L���A��*.

	loss/loss��>

accuracy/accuracy_1qfw?���4=       `I��	�.TN���A��*.

	loss/loss��>

accuracy/accuracy_1�pw?'�=       `I��	G�{P���A��*.

	loss/loss�x�>

accuracy/accuracy_19\w?[1�=       `I��	z#�R���A��*.

	loss/loss�_�>

accuracy/accuracy_1%)w?���=       `I��	�LU���A��*.

	loss/lossV��>

accuracy/accuracy_1��v?�ZT�=       `I��	�ϩW���A��*.

	loss/loss���>

accuracy/accuracy_1
w?G�t=       `I��	���Y���A��*.

	loss/loss���>

accuracy/accuracy_1�v?�z�=       `I��	���[���A��*.

	loss/lossV��>

accuracy/accuracy_1بw?��~�=       `I��	CtR^���A��*.

	loss/lossO��>

accuracy/accuracy_1qfw?,�|�=       `I��	$�`���A��*.

	loss/loss���>

accuracy/accuracy_1Rw?&(3=       `I��	<@�b���A��*.

	loss/lossk��>

accuracy/accuracy_1Uaw?�TU�=       `I��	$�d���A��*.

	loss/loss=��>

accuracy/accuracy_1Rw?`u�=       `I��	$�bg���A��*.

	loss/loss�}�>

accuracy/accuracy_1�\v?��]�=       `I��	��i���A��*.

	loss/lossC�>

accuracy/accuracy_1�hs?��,=       `I��	+l���A��*.

	loss/lossy�>

accuracy/accuracy_1�9u?2�I=       `I��	I�Vn���A��*.

	loss/lossE�>

accuracy/accuracy_1��v?k���=       `I��	0��p���A��*.

	loss/loss�C�>

accuracy/accuracy_1��w?��I=       `I��	J�0s���A��*.

	loss/loss"ݾ>

accuracy/accuracy_1�uw?��,=       `I��	���u���A��*.

	loss/loss^ο>

accuracy/accuracy_1�pw?�tn�=       `I��	VG�w���A��*.

	loss/loss`Ͼ>

accuracy/accuracy_1�w?����=       `I��	tY�y���A��*.

	loss/loss��>

accuracy/accuracy_1Ww?��TR=       `I��	��Z|���A��*.

	loss/loss�x�>

accuracy/accuracy_12�w?tv
%=       `I��	O��~���A��*.

	loss/loss*��>

accuracy/accuracy_12�w?�ū�=       `I��	�0����A��*.

	loss/loss�X�>

accuracy/accuracy_1�w?��#
=       `I��	������A��*.

	loss/lossuM�>

accuracy/accuracy_1�uw?�!��=       `I��	#�����A��*.

	loss/loss�(�>

accuracy/accuracy_1�w?�D�=       `I��	쇾��A��*.

	loss/loss6�>

accuracy/accuracy_1+�w?�,=       `I��	'�X����A��*.

	loss/lossm�>

accuracy/accuracy_1�zw?���`=       `I��	������A��*.

	loss/lossf��>

accuracy/accuracy_1��w?�$7=       `I��	����A��*.

	loss/loss�5�>

accuracy/accuracy_12�w?�O0�=       `I��	�1����A��*.

	loss/loss�7�>

accuracy/accuracy_1��v?2��=       `I��	�~����A��*.

	loss/loss=��>

accuracy/accuracy_1Rw?��=       `I��	!�ꕾ��A��*.

	loss/loss��>

accuracy/accuracy_1
w?4֫R=       `I��	WBח���A��*.

	loss/lossR��>

accuracy/accuracy_1Uaw?<_�D=       `I��	�:����A��*.

	loss/lossNm�>

accuracy/accuracy_1Ww?x���=       `I��	�Ш����A��*.

	loss/loss$�>

accuracy/accuracy_1Rw?��=       `I��	������A��*.

	loss/loss$��>

accuracy/accuracy_1!�s?z'5=       `I��	��C����A��*.

	loss/lossoc�>

accuracy/accuracy_1�\v?��OP=       `I��	��_����A��*.

	loss/loss��>

accuracy/accuracy_1
w?�\�=       `I��	:6Υ���A��*.

	loss/loss���>

accuracy/accuracy_1@.w?ebn�=       `I��	�=����A��*.

	loss/loss踺>

accuracy/accuracy_1�kw?%��=       `I��	y�����A��*.

	loss/loss,�>

accuracy/accuracy_1x?`���=       `I��	�ޙ����A��*.

	loss/loss��>

accuracy/accuracy_1�	x?���=       `I��	r�����A��*.

	loss/loss��>

accuracy/accuracy_1Tx?xw�=       `I��		e����A�*.

	loss/loss6Y�>

accuracy/accuracy_1�7x?Ts��=       `I��	mvӳ���A̂*.

	loss/loss��>

accuracy/accuracy_1�2x?q�8�=       `I��	�)����A��*.

	loss/lossc��>

accuracy/accuracy_1بw?���=       `I��	�����A��*.

	loss/loss�q�>

accuracy/accuracy_12�w?��=       `I��	�噺���A��*.

	loss/loss���>

accuracy/accuracy_1[�w?5,�=       `I��	C�	����A܅*.

	loss/loss�>

accuracy/accuracy_1��w?�?J=       `I��	�au����A��*.

	loss/loss/��>

accuracy/accuracy_1��w?<�&q=       `I��	̬f����A��*.

	loss/loss*P�>

accuracy/accuracy_1w�w? ��+=       `I��	'v�þ��A��*.

	loss/loss�޾>

accuracy/accuracy_1[�w?�,#!=       `I��	�,ƾ��A�*.

	loss/lossm׻>

accuracy/accuracy_1Tx?ڐV�=       `I��	9�Ⱦ��AЉ*.

	loss/lossg��>

accuracy/accuracy_1��w?���=       `I��	ǅ�ʾ��A��*.

	loss/loss���>

accuracy/accuracy_1��w?��'�=       `I��	d��̾��A��*.

	loss/loss:�>

accuracy/accuracy_1�Gw?�T�=       `I��	pMMϾ��A��*.

	loss/loss�~�>

accuracy/accuracy_1��w?�
=       `I��	��Ѿ��A��*.

	loss/loss%��>

accuracy/accuracy_1�pw?�<�=       `I��		>%Ծ��Ač*.

	loss/loss��>

accuracy/accuracy_1��w?P:�L=       `I��	��־��A��*.

	loss/loss�V�>

accuracy/accuracy_1�w?�4�=       `I��	��pؾ��A��*.

	loss/lossp��>

accuracy/accuracy_1�qu?`M.�=       `I��	��ھ��A��*.

	loss/loss���>

accuracy/accuracy_1�w?\�}5=       `I��	�Nݾ��AԐ*.

	loss/loss+R�>

accuracy/accuracy_1x?N�=Y=       `I��	^��߾��A��*.

	loss/loss��>

accuracy/accuracy_1~�w?zH��=       `I��	���ᾒ�A��*.

	loss/lossk��>

accuracy/accuracy_1MLx?���=       `I��	
e;侒�A��*.

	loss/loss*¸>

accuracy/accuracy_1MLx?K� �=       `I��	��澒�A�*.

	loss/loss�>�>

accuracy/accuracy_1p#x?�@�=       `I��	�龒�AȔ*.

	loss/lossy�>

accuracy/accuracy_11Gx?F�
�=       `I��	��뾒�A��*.

	loss/loss���>

accuracy/accuracy_11Gx?��0�=       `I��	<h\����A��*.

	loss/loss Y�>

accuracy/accuracy_1�`x?a��=       `I��	ul�ﾒ�A��*.

	loss/loss0��>

accuracy/accuracy_1�ex?��{�=       `I��	4s2��Aؗ*.

	loss/loss�c�>

accuracy/accuracy_1x?��r�=       `I��	��h����A��*.

	loss/loss���>

accuracy/accuracy_1x?�!.&=       `I��	��~����A��*.

	loss/loss��>

accuracy/accuracy_1�<x?]7i=       `I��	�������A��*.

	loss/loss/޽>

accuracy/accuracy_1x?�.?=       `I��	��N����A�*.

	loss/lossu@�>

accuracy/accuracy_19\w?�d�=       `I��	ы�����A̛*.

	loss/loss�8�>

accuracy/accuracy_1�2x?d�
�=       `I��	�_�����A��*.

	loss/loss��>

accuracy/accuracy_1x?HՋ=       `I��	ڨ����A��*.

	loss/loss ��>

accuracy/accuracy_1�w?�c=       `I��	f���A��*.

	loss/loss���>

accuracy/accuracy_1x?<�n=       `I��	P�����Aܞ*.

	loss/loss�$?

accuracy/accuracy_1�_q?��)=       `I��	G�	���A��*.

	loss/loss>��>

accuracy/accuracy_1Nu?���?=       `I��	����A��*.

	loss/lossJ��>

accuracy/accuracy_1ĸv?f�.�=       `I��	�{���A��*.

	loss/loss�ֻ>

accuracy/accuracy_1$�w?�&w�=       `I��	Ś����A�*.

	loss/loss�Ӻ>

accuracy/accuracy_11Gx?]���=       `I��	#.J���AТ*.

	loss/loss`��>

accuracy/accuracy_1�x?lX��=       `I��	r�H���A��*.

	loss/loss���>

accuracy/accuracy_1?�x?]�7c=       `I��	�����A��*.

	loss/loss���>

accuracy/accuracy_1��x?�5\W=       `I��	Xq����A��*.

	loss/loss��>

accuracy/accuracy_1��x?YJW==       `I��	]%`���A�*.

	loss/loss֮�>

accuracy/accuracy_1o�x?y}��=       `I��	����AĦ*.

	loss/lossn|�>

accuracy/accuracy_1��x?��_g=       `I��	,�����A��*.

	loss/loss�/�>

accuracy/accuracy_1�`x?�^�*=       `I��	52"���A��*.

	loss/lossߏ�>

accuracy/accuracy_1Bx?�k	�=       `I��	1�q$���A�*.

	loss/lossP��>

accuracy/accuracy_1[�x?A7,=       `I��	���&���Aԩ*.

	loss/lossdQ�>

accuracy/accuracy_1#�x?� u�=       `I��	���(���A��*.

	loss/loss��>

accuracy/accuracy_1Гx?1�8=       `I��	;�!+���A��*.

	loss/loss�v�>

accuracy/accuracy_1�`x?�\o=       `I��	cL�-���A��*.

	loss/loss?1�>

accuracy/accuracy_1Гx?K��=       `I��	޹�/���A�*.

	loss/lossj%�>

accuracy/accuracy_1Fzx?z�&5=       `I��	Z�,2���Aȭ*.

	loss/loss��>

accuracy/accuracy_1px?8<u�=       `I��	ja64���A��*.

	loss/loss�<�>

accuracy/accuracy_1[�w?1�=       `I��	vW�6���A��*.

	loss/loss�J�>

accuracy/accuracy_1�It?���"=       `I��	��9���A��*.

	loss/loss\+�>

accuracy/accuracy_1r�u?f4-b=       `I��	 m;���Aذ*.

	loss/loss��>

accuracy/accuracy_11Gx?��=       `I��	��o=���A��*.

	loss/loss<~�>

accuracy/accuracy_1��x?�$�=       `I��	a�?���A��*.

	loss/lossǲ>

accuracy/accuracy_1}�x?���=       `I��	�  B���A��*.

	loss/lossd��>

accuracy/accuracy_1?�x?�[=       `I��	�Y�D���A�*.

	loss/loss�>

accuracy/accuracy_1��x?�Z��=       `I��	�W�F���A̴*.

	loss/loss{�>

accuracy/accuracy_1o�x?�i��=       `I��	]G�H���A��*.

	loss/loss��>

accuracy/accuracy_1�x?S�
=       `I��	N!6K���A��*.

	loss/lossZó>

accuracy/accuracy_1S�x?Z8��=       `I��	S-�M���A��*.

	loss/loss��>

accuracy/accuracy_1L	y?LN��=       `I��	!PP���Aܷ*.

	loss/lossO͹>

accuracy/accuracy_1[�x?X�x�=       `I��	hT	R���A��*.

	loss/loss�	�>

accuracy/accuracy_1�x?�F0=       `I��	d�T���A��*.

	loss/loss諵>

accuracy/accuracy_1S�x?�eG=       `I��	��W���A��*.

	loss/loss$�>

accuracy/accuracy_1��x?y�=       `I��	[|Y���A�*.

	loss/loss��>

accuracy/accuracy_1 �x?����=       `I��	Ri�[���Aл*.

	loss/loss�G�>

accuracy/accuracy_1��x?h���=       `I��	Ɯ�]���A��*.

	loss/loss��>

accuracy/accuracy_1بw?$���=       `I��	7�8`���A��*.

	loss/lossl�>

accuracy/accuracy_1j�v?�m=       `I��	3��b���A��*.

	loss/loss`��>

accuracy/accuracy_1j�v?^!=       `I��	�he���A�*.

	loss/loss�>

accuracy/accuracy_1b�w?�;��=       `I��	��g���AĿ*.

	loss/loss�_�>

accuracy/accuracy_1bx?ɞ��=       `I��	V�Xi���A��*.

	loss/loss��>

accuracy/accuracy_1�[x?�W&�=       `I��	Z��k���A��*.

	loss/loss��>

accuracy/accuracy_1�y?9��=       `I��	p�6n���A��*.

	loss/loss�>

accuracy/accuracy_1��x?AM#=       `I��	ܳ�p���A��*.

	loss/loss"�>

accuracy/accuracy_1�y?T��=       `I��	7��r���A��*.

	loss/loss�۫>

accuracy/accuracy_1*2y?**�Q=       `I��	���t���A��*.

	loss/lossYҩ>

accuracy/accuracy_1�Py?��,-=       `I��	�s^w���A��*.

	loss/loss�֩>

accuracy/accuracy_1 �y?���b=       `I��	j��y���A��*.

	loss/loss��>

accuracy/accuracy_1�Fy?��/�=       `I��	�@�{���A��*.

	loss/lossKn�>

accuracy/accuracy_1"`y?�i=       `I��	�h~���A��*.

	loss/loss�$�>

accuracy/accuracy_1�Py?ߚt=       `I��	fB~����A��*.

	loss/loss�;�>

accuracy/accuracy_1�'y?���=       `I��	��킿��A��*.

	loss/lossͅ�>

accuracy/accuracy_1�y?rY�=       `I��	�+:����A��*.

	loss/loss��>

accuracy/accuracy_1�x?L�1=       `I��	#:����A��*.

	loss/loss&�>

accuracy/accuracy_1��x?�hd�=       `I��	�������A��*.

	loss/lossd/�>

accuracy/accuracy_1cw?�4w=       `I��	�,����A��*.

	loss/loss�q�>

accuracy/accuracy_1��v?��Wf=       `I��	IX�����A��*.

	loss/loss�)�>

accuracy/accuracy_1�(x?_�ހ=       `I��	������A��*.

	loss/loss��>

accuracy/accuracy_11y?K�e�=       `I��	��В���A��*.

	loss/losss��>

accuracy/accuracy_1a<y?�7�=       `I��	Wp@����A��*.

	loss/loss���>

accuracy/accuracy_1Zjy?���=       `I��	XE�����A��*.

	loss/loss̤>

accuracy/accuracy_17�y?�~$k=       `I��	�f�����A��*.

	loss/lossA�>

accuracy/accuracy_1�ty?��1B=       `I��	������A��*.

	loss/lossM�>

accuracy/accuracy_1 �y?S�r=       `I��	�k����A��*.

	loss/losso$�>

accuracy/accuracy_1�y?T[��=       `I��	�3頿��A��*.

	loss/lossXx�>

accuracy/accuracy_1n�y?�&�=       `I��	2W����A��*.

	loss/loss�,�>

accuracy/accuracy_1�Uy?J��=       `I��	�_f����A��*.

	loss/loss���>

accuracy/accuracy_1uoy?z`�=       `I��	ָ�����A��*.

	loss/loss�N�>

accuracy/accuracy_1�ty?/<�=       `I��	av����A��*.

	loss/lossfЧ>

accuracy/accuracy_1�Py?��=       `I��	Ou����A��*.

	loss/loss���>

accuracy/accuracy_1L	y?UPe=       `I��	�@Į���A��*.

	loss/loss\��>

accuracy/accuracy_1*2y?xK)6=       `I��	-������A��*.

	loss/loss��>

accuracy/accuracy_1L	y?�۶q=       `I��	�+����A��*.

	loss/loss*O�>

accuracy/accuracy_1>ey?�H��=       `I��	�)�����A��*.

	loss/loss���>

accuracy/accuracy_1[y?Jwd=       `I��	ە
����A��*.

	loss/loss�?

accuracy/accuracy_1��s?P�Jd=       `I��	������A��*.

	loss/loss.�>

accuracy/accuracy_1�2x?e)[�=       `I��	aEX����A��*.

	loss/loss�:�>

accuracy/accuracy_1N�w?fUk=       `I��	'ž���A��*.

	loss/loss~?�>

accuracy/accuracy_1�x?����=       `I��	�0����A��*.

	loss/losskP�>

accuracy/accuracy_1�x?j͔�=       `I��	4��ÿ��A��*.

	loss/loss�z�>

accuracy/accuracy_1�Uy?�J��=       `I��	P�~ſ��A��*.

	loss/lossc�>

accuracy/accuracy_1L�y?�B=       `I��	��#ȿ��A��*.

	loss/loss���>

accuracy/accuracy_17�y?�)=       `I��	~9�ʿ��A��*.

	loss/lossn��>

accuracy/accuracy_1L�y?��n=       `I��	�;�̿��A��*.

	loss/loss�!�>

accuracy/accuracy_1g�y?z.��=       `I��	��Ͽ��A��*.

	loss/loss�-�>

accuracy/accuracy_1`�y?���=       `I��	M5ѿ��A��*.

	loss/loss��>

accuracy/accuracy_1��y?�Ꮣ=       `I��	wɛӿ��A��*.

	loss/loss��>

accuracy/accuracy_1�y?�P=       `I��	q(ֿ��A��*.

	loss/lossb�>

accuracy/accuracy_1��y?t���=       `I��	��Pؿ��A��*.

	loss/loss/դ>

accuracy/accuracy_1�yy?�I{.=       `I��	� Kڿ��A��*.

	loss/loss�Þ>

accuracy/accuracy_1Y'z?L�=       `I��	0-�ܿ��A��*.

	loss/loss�)�>

accuracy/accuracy_1)�y?�I� =       `I��	��߿��A��*.

	loss/loss�S�>

accuracy/accuracy_1[y?�c��=       `I��	�~�ῒ�A��*.

	loss/loss'V�>

accuracy/accuracy_1a<y?���=       `I��	i�㿒�A��*.

	loss/loss�C�>

accuracy/accuracy_1 �y?.��[=       `I��	��忒�A��*.

	loss/loss�O�>

accuracy/accuracy_1v�x?��0=       `I��	��7迒�A��*.

	loss/loss��>

accuracy/accuracy_1y�u?�9�=       `I��	��꿒�A��*.

	loss/loss�1�>

accuracy/accuracy_1�Bw?Ԝ�=       `I��	fj����A��*.

	loss/loss���>

accuracy/accuracy_1��x?�Y��=       `I��	�T��A��*.

	loss/lossHy�>

accuracy/accuracy_1n�y?�BJ=       `I��	��a��A��*.

	loss/lossU��>

accuracy/accuracy_1�z?��G=       `I��	/����A��*.

	loss/lossѯ�>

accuracy/accuracy_1�y?b��=       `I��	�*����A��*.

	loss/losss��>

accuracy/accuracy_1��y?�JM=       `I��	aXD����A��*.

	loss/loss��>

accuracy/accuracy_1��y?��=b=       `I��	̚v����A��*.

	loss/loss;�>

accuracy/accuracy_1n�y?q"r�=       `I��	Z������A��*.

	loss/loss�߲>

accuracy/accuracy_1Zjy?���\=       `I��	-�G����A��*.

	loss/loss�}�>

accuracy/accuracy_1`�y?K=       `I��	�e����A��*.

	loss/loss3|�>

accuracy/accuracy_1)�y?��#=       `I��	9J����A��*.

	loss/loss֬>

accuracy/accuracy_1|�y?��=       `I��	^?����A��*.

	loss/lossM̳>

accuracy/accuracy_17�y?�K�=       `I��	�Y���A��*.

	loss/lossq�>

accuracy/accuracy_1S�y?��Ɔ=       `I��	x�
���A��*.

	loss/loss{��>

accuracy/accuracy_1L�y?S�\�=       `I��	������A��*.

	loss/loss�,�>

accuracy/accuracy_1��y?�]�=       `I��	�t���A��*.

	loss/lossqЯ>

accuracy/accuracy_1[y?�n�S=       `I��	�l���A��*.

	loss/loss���>

accuracy/accuracy_1��y?���$=       `I��	�����A��*.

	loss/loss�T�>

accuracy/accuracy_1&lv?�Q��=       `I��	G{'���A��*.

	loss/loss��>

accuracy/accuracy_1�w?�^�J=       `I��	�{���A��*.

	loss/loss0��>

accuracy/accuracy_1�w?�b =       `I��	��y���A��*.

	loss/loss ��>

accuracy/accuracy_1�y?���=       `I��	)�����A��*.

	loss/lossBˬ>

accuracy/accuracy_1 �y?��E=       `I��	-5G���A��*.

	loss/loss��>

accuracy/accuracy_1[y?{N�=       `I��	f6b!���A��*.

	loss/lossxe�>

accuracy/accuracy_1D�y?8F	j=       `I��	,��#���A��*.

	loss/lossBʯ>

accuracy/accuracy_1��y?��y=       `I��	\f�%���A��*.

	loss/loss���>

accuracy/accuracy_1�z?��=       `I��	�\(���A��*.

	loss/loss4��>

accuracy/accuracy_1�1z?���=       `I��	.[�*���A��*.

	loss/loss�ެ>

accuracy/accuracy_1z?�=       `I��	���,���A܂*.

	loss/loss5ұ>

accuracy/accuracy_1��y?.���=       `I��	Ό/���A��*.

	loss/loss�ʴ>

accuracy/accuracy_17�y?���=       `I��	?w1���A��*.

	loss/lossG�>

accuracy/accuracy_1g�y?���f=       `I��	;��3���A��*.

	loss/loss潬>

accuracy/accuracy_1u,z?�ٰ}=       `I��	<�6���A�*.

	loss/loss�j�>

accuracy/accuracy_1 �y?�ۙ:=       `I��	�a*8���AІ*.

	loss/lossa�>

accuracy/accuracy_1[y?�Ǟ-