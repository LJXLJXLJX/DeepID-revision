       �K"	  ��ȑ�Abrain.Event:2]��     ����	|c��ȑ�A"ǳ
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
initNoOp%^Conv_layer_1/weights/Variable/Assign$^Conv_layer_1/biases/Variable/Assign%^Conv_layer_2/weights/Variable/Assign$^Conv_layer_2/biases/Variable/Assign%^Conv_layer_3/weights/Variable/Assign$^Conv_layer_3/biases/Variable/Assign%^Conv_layer_4/weights/Variable/Assign$^Conv_layer_4/biases/Variable/Assign ^DeepID1/weights/Variable/Assign"^DeepID1/weights_1/Variable/Assign^DeepID1/biases/Variable/Assign&^loss/nn_layer/weights/Variable/Assign%^loss/nn_layer/biases/Variable/Assign^train/beta1_power/Assign^train/beta2_power/Assign*^Conv_layer_1/weights/Variable/Adam/Assign,^Conv_layer_1/weights/Variable/Adam_1/Assign)^Conv_layer_1/biases/Variable/Adam/Assign+^Conv_layer_1/biases/Variable/Adam_1/Assign*^Conv_layer_2/weights/Variable/Adam/Assign,^Conv_layer_2/weights/Variable/Adam_1/Assign)^Conv_layer_2/biases/Variable/Adam/Assign+^Conv_layer_2/biases/Variable/Adam_1/Assign*^Conv_layer_3/weights/Variable/Adam/Assign,^Conv_layer_3/weights/Variable/Adam_1/Assign)^Conv_layer_3/biases/Variable/Adam/Assign+^Conv_layer_3/biases/Variable/Adam_1/Assign*^Conv_layer_4/weights/Variable/Adam/Assign,^Conv_layer_4/weights/Variable/Adam_1/Assign)^Conv_layer_4/biases/Variable/Adam/Assign+^Conv_layer_4/biases/Variable/Adam_1/Assign%^DeepID1/weights/Variable/Adam/Assign'^DeepID1/weights/Variable/Adam_1/Assign'^DeepID1/weights_1/Variable/Adam/Assign)^DeepID1/weights_1/Variable/Adam_1/Assign$^DeepID1/biases/Variable/Adam/Assign&^DeepID1/biases/Variable/Adam_1/Assign+^loss/nn_layer/weights/Variable/Adam/Assign-^loss/nn_layer/weights/Variable/Adam_1/Assign*^loss/nn_layer/biases/Variable/Adam/Assign,^loss/nn_layer/biases/Variable/Adam_1/Assign";���     ��0�	����ȑ�AJ��
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


train/Adamy���9       �7�	:T��ȑ�A*.

	loss/loss
9D

accuracy/accuracy_1��`:[oI�;       #�\	�*Iɑ�Ad*.

	loss/loss���@

accuracy/accuracy_1s/u:��*<       ȷ�R	?�Qɑ�A�*.

	loss/lossS��@

accuracy/accuracy_1s/u:F�u�<       ȷ�R	I��ɑ�A�*.

	loss/lossi��@

accuracy/accuracy_1��`:���<       ȷ�R	73�ɑ�A�*.

	loss/loss��@

accuracy/accuracy_15RL:憋�<       ȷ�R	.vWɑ�A�*.

	loss/loss=��@

accuracy/accuracy_1	�;�
�<       ȷ�R	�oɑ�A�*.

	loss/lossn��@

accuracy/accuracy_1PY;PT�<       ȷ�R	�$�ɑ�A�*.

	loss/loss�e�@

accuracy/accuracy_1#�j;�ji0<       ȷ�R	�m�ɑ�A�*.

	loss/lossC�@

accuracy/accuracy_1�\�;�̛z<       ȷ�R	(��ɑ�A�*.

	loss/lossM�@

accuracy/accuracy_1�x�;@n��<       ȷ�R	��Xɑ�A�*.

	loss/loss3��@

accuracy/accuracy_1G��;���s<       ȷ�R	M"�ɑ�A�*.

	loss/losse��@

accuracy/accuracy_1:�;<n<       ȷ�R	E ɑ�A�	*.

	loss/loss���@

accuracy/accuracy_1�m�;#���<       ȷ�R	&��"ɑ�A�
*.

	loss/loss�m�@

accuracy/accuracy_1��;��2�<       ȷ�R	Vj4%ɑ�A�
*.

	loss/loss�/�@

accuracy/accuracy_1s/�;&�u�<       ȷ�R	8��'ɑ�A�*.

	loss/lossJ��@

accuracy/accuracy_1ǣ<��z<       ȷ�R	��$*ɑ�A�*.

	loss/loss���@

accuracy/accuracy_1�1<�+��<       ȷ�R	�,ɑ�A�*.

	loss/loss���@

accuracy/accuracy_1�$<�OyF<       ȷ�R	C�.ɑ�A�*.

	loss/loss[�@

accuracy/accuracy_1��)<����<       ȷ�R	쌞1ɑ�A�*.

	loss/loss�Z�@

accuracy/accuracy_1��)<�fG�<       ȷ�R	1�4ɑ�A�*.

	loss/loss���@

accuracy/accuracy_1��2<p}E<       ȷ�R	�<�6ɑ�A�*.

	loss/loss���@

accuracy/accuracy_1��@<q��t<       ȷ�R	�@9ɑ�A�*.

	loss/lossˢ�@

accuracy/accuracy_1��D<�p��<       ȷ�R	9X<ɑ�A�*.

	loss/loss~;�@

accuracy/accuracy_1�6G<�߻N<       ȷ�R	� �>ɑ�A�*.

	loss/loss��@

accuracy/accuracy_15RL<:�`�<       ȷ�R	�~Aɑ�A�*.

	loss/lossn��@

accuracy/accuracy_1	�N<K?��<       ȷ�R	vJuCɑ�A�*.

	loss/loss˔�@

accuracy/accuracy_1��d<�s�S<       ȷ�R	���Eɑ�A�*.

	loss/loss���@

accuracy/accuracy_1��s<-�,<       ȷ�R	�x�Hɑ�A�*.

	loss/loss���@

accuracy/accuracy_11y<�t6<       ȷ�R	��JKɑ�A�*.

	loss/lossZX�@

accuracy/accuracy_15A�<�n�<       ȷ�R	L�|Mɑ�A�*.

	loss/loss(��@

accuracy/accuracy_1R �<�T��<       ȷ�R	|�Pɑ�A�*.

	loss/loss�y�@

accuracy/accuracy_1BM�<!�̭<       ȷ�R	?��Rɑ�A�*.

	loss/loss"�@

accuracy/accuracy_1���<ph�<       ȷ�R	��VUɑ�A�*.

	loss/loss���@

accuracy/accuracy_1PY�<�_�<       ȷ�R	���Wɑ�A�*.

	loss/loss���@

accuracy/accuracy_1(�<v�S�<       ȷ�R	2�Yɑ�A�*.

	loss/loss���@

accuracy/accuracy_1�ש<1�#�<       ȷ�R	כ{\ɑ�A�*.

	loss/loss�5�@

accuracy/accuracy_1y$�<�4�<       ȷ�R	�_ɑ�A�*.

	loss/loss;��@

accuracy/accuracy_1.L�<Z�v3<       ȷ�R	���aɑ�A�*.

	loss/loss��@

accuracy/accuracy_1��<���<       ȷ�R	�W�cɑ�A�*.

	loss/loss���@

accuracy/accuracy_1,��<��!<       ȷ�R	$Dfɑ�A�*.

	loss/loss��@

accuracy/accuracy_1<X�<DY΃<       ȷ�R	h�hɑ�A� *.

	loss/lossMa�@

accuracy/accuracy_1���<Yg�<       ȷ�R	���kɑ�A� *.

	loss/loss��@

accuracy/accuracy_1?�<�qc<       ȷ�R	L_ nɑ�A�!*.

	loss/lossP��@

accuracy/accuracy_1@��<�_<       ȷ�R	j�-pɑ�A�"*.

	loss/loss]i�@

accuracy/accuracy_1N}=D1�<       ȷ�R	��rɑ�A�#*.

	loss/lossk��@

accuracy/accuracy_1h�=O���<       ȷ�R	�vɑ�A�#*.

	loss/loss�|�@

accuracy/accuracy_1�1=���<       ȷ�R	��5yɑ�A�$*.

	loss/loss���@

accuracy/accuracy_1�=�R��<       ȷ�R	h*�{ɑ�A�%*.

	loss/loss���@

accuracy/accuracy_1=##=�T��<       ȷ�R	��6~ɑ�A�&*.

	loss/lossPC�@

accuracy/accuracy_1J/1=���<       ȷ�R	�oL�ɑ�A�'*.

	loss/loss%��@

accuracy/accuracy_1�f;=��l<       ȷ�R	4�τɑ�A�'*.

	loss/lossMC�@

accuracy/accuracy_1�lB=�e��<       ȷ�R	j��ɑ�A�(*.

	loss/lossp�@

accuracy/accuracy_1{ L=H8X<       ȷ�R	��ɑ�A�)*.

	loss/loss���@

accuracy/accuracy_1iY=��ӣ<       ȷ�R	n���ɑ�A�**.

	loss/lossQ��@

accuracy/accuracy_13�b=�sڴ<       ȷ�R	 r�ɑ�A�**.

	loss/loss��@

accuracy/accuracy_1Vpo=��g.<       ȷ�R	|��ɑ�A�+*.

	loss/loss{�@

accuracy/accuracy_1c|}=pD"�<       ȷ�R	�(Öɑ�A�,*.

	loss/loss���@

accuracy/accuracy_1h��=Y�f�<       ȷ�R	Kզ�ɑ�A�-*.

	loss/loss>��@

accuracy/accuracy_1��=HP9�<       ȷ�R	�?ќɑ�A�.*.

	loss/loss�}�@

accuracy/accuracy_1�y�=(x�<       ȷ�R	ĕ��ɑ�A�.*.

	loss/loss'��@

accuracy/accuracy_1۵�=�zb+<       ȷ�R	� ��ɑ�A�/*.

	loss/loss=��@

accuracy/accuracy_1y6�=I���<       ȷ�R	���ɑ�A�0*.

	loss/loss���@

accuracy/accuracy_1:�=u[	 <       ȷ�R	� �ɑ�A�1*.

	loss/loss
3�@

accuracy/accuracy_1�=�=G٠<       ȷ�R	��/�ɑ�A�2*.

	loss/loss	D�@

accuracy/accuracy_1�C�=8(:�<       ȷ�R	��H�ɑ�A�2*.

	loss/loss��@

accuracy/accuracy_1���=�9<       ȷ�R	L�Űɑ�A�3*.

	loss/loss��@

accuracy/accuracy_1c�=^�)<       ȷ�R	ʘp�ɑ�A�4*.

	loss/loss!�@

accuracy/accuracy_1�H�=_�)<       ȷ�R	��u�ɑ�A�5*.

	loss/lossߡ�@

accuracy/accuracy_1i��=�
E�<       ȷ�R	�X��ɑ�A�5*.

	loss/loss\{�@

accuracy/accuracy_1E�=?�L�<       ȷ�R	��t�ɑ�A�6*.

	loss/loss�#�@

accuracy/accuracy_1ͻ>шO<       ȷ�R	n�b�ɑ�A�7*.

	loss/lossj��@

accuracy/accuracy_1� >8Ҋ�<       ȷ�R	�9��ɑ�A�8*.

	loss/lossQI�@

accuracy/accuracy_1<G	>�rk<       ȷ�R	qz��ɑ�A�9*.

	loss/loss�ǰ@

accuracy/accuracy_1\�>�+�<       ȷ�R	����ɑ�A�9*.

	loss/loss��@

accuracy/accuracy_1�>��c<       ȷ�R	�5�ɑ�A�:*.

	loss/lossɩ�@

accuracy/accuracy_1�� >�P��<       ȷ�R	W��ɑ�A�;*.

	loss/lossT��@

accuracy/accuracy_17)>|��<       ȷ�R	Qc��ɑ�A�<*.

	loss/loss���@

accuracy/accuracy_1�%0>�$<       ȷ�R	�8�ɑ�A�<*.

	loss/loss�m�@

accuracy/accuracy_1�+7>
�1H<       ȷ�R	�F�ɑ�A�=*.

	loss/loss�)�@

accuracy/accuracy_1�CB>�F�<       ȷ�R	��4�ɑ�A�>*.

	loss/loss��@

accuracy/accuracy_1�J>)���<       ȷ�R	��4�ɑ�A�?*.

	loss/loss[��@

accuracy/accuracy_1��P>�d:E<       ȷ�R	AO�ɑ�A�@*.

	loss/lossBg�@

accuracy/accuracy_1�[>���<       ȷ�R	�Sb�ɑ�A�@*.

	loss/loss�Q�@

accuracy/accuracy_1�Y>n\�<       ȷ�R	mY��ɑ�A�A*.

	loss/loss�0�@

accuracy/accuracy_1#�j>��T�<       ȷ�R	��J�ɑ�A�B*.

	loss/lossÖ�@

accuracy/accuracy_1��s>��|�<       ȷ�R	�Wn�ɑ�A�C*.

	loss/lossF��@

accuracy/accuracy_1$]~>!I�s<       ȷ�R	_
y�ɑ�A�C*.

	loss/loss΍�@

accuracy/accuracy_1���>����<       ȷ�R	���ɑ�A�D*.

	loss/loss7x�@

accuracy/accuracy_1�o�>e��<       ȷ�R	x$��ɑ�A�E*.

	loss/loss��@

accuracy/accuracy_1��>��)<       ȷ�R	G��ɑ�A�F*.

	loss/loss���@

accuracy/accuracy_1+�>l>*�<       ȷ�R	Z���ɑ�A�G*.

	loss/losscr�@

accuracy/accuracy_1�p�>�q��<       ȷ�R	���ɑ�A�G*.

	loss/loss�ʄ@

accuracy/accuracy_1�L�>Fa�<       ȷ�R	�pU�ɑ�A�H*.

	loss/loss�i�@

accuracy/accuracy_1��>���]<       ȷ�R	y%��ɑ�A�I*.

	loss/loss��~@

accuracy/accuracy_1�1�>L�,@<       ȷ�R	���ʑ�A�J*.

	loss/lossORx@

accuracy/accuracy_1"�>�/�<       ȷ�R	�'ʑ�A�K*.

	loss/loss>q@

accuracy/accuracy_1�o�>���<       ȷ�R	<r	ʑ�A�K*.

	loss/loss�l@

accuracy/accuracy_1���>#�^�<       ȷ�R	���
ʑ�A�L*.

	loss/loss��f@

accuracy/accuracy_1���>�X�<       ȷ�R	I��ʑ�A�M*.

	loss/lossI�a@

accuracy/accuracy_1~��>;+b1<       ȷ�R	��ʑ�A�N*.

	loss/loss�X@

accuracy/accuracy_1<��>t��<       ȷ�R	I"ʑ�A�N*.

	loss/loss��S@

accuracy/accuracy_1�>���<       ȷ�R	,��ʑ�A�O*.

	loss/loss�N@

accuracy/accuracy_1Ɠ�>3�6�<       ȷ�R	"h�ʑ�A�P*.

	loss/lossߥE@

accuracy/accuracy_1��>���<       ȷ�R	�juʑ�A�Q*.

	loss/loss�@@

accuracy/accuracy_1k�>v�<       ȷ�R	;�ʑ�A�R*.

	loss/loss�9@

accuracy/accuracy_1Y� ?��4<       ȷ�R	�R�"ʑ�A�R*.

	loss/lossj5@

accuracy/accuracy_1I1?�"i�<       ȷ�R	��$ʑ�A�S*.

	loss/loss�0@

accuracy/accuracy_1�b?@>k�<       ȷ�R	��l'ʑ�A�T*.

	loss/loss�)@

accuracy/accuracy_1��	?C*:�<       ȷ�R	>k�*ʑ�A�U*.

	loss/loss �"@

accuracy/accuracy_1�?I8X�<       ȷ�R	�3.ʑ�A�U*.

	loss/loss�@

accuracy/accuracy_1�e?��<       ȷ�R	�h�0ʑ�A�V*.

	loss/loss n@

accuracy/accuracy_1�@?4vS�<       ȷ�R	Q@�2ʑ�A�W*.

	loss/loss�?@

accuracy/accuracy_1k?H�>�<       ȷ�R	�6ʑ�A�X*.

	loss/lossm�@

accuracy/accuracy_1�?'�-�<       ȷ�R	�0�8ʑ�A�Y*.

	loss/loss��	@

accuracy/accuracy_1��?�J�<       ȷ�R	���;ʑ�A�Y*.

	loss/lossLc@

accuracy/accuracy_1hR!?4\N^<       ȷ�R	d\>ʑ�A�Z*.

	loss/loss�}@

accuracy/accuracy_1V_%?�GQ3<       ȷ�R	��@ʑ�A�[*.

	loss/loss�;�?

accuracy/accuracy_11�'?̀�k<       ȷ�R	c�BDʑ�A�\*.

	loss/loss���?

accuracy/accuracy_1�,?J�'+<       ȷ�R	��VGʑ�A�\*.

	loss/loss���?

accuracy/accuracy_1]e,?��<       ȷ�R	�'�Iʑ�A�]*.

	loss/loss��?

accuracy/accuracy_1�|0?L�p�<       ȷ�R	6�CLʑ�A�^*.

	loss/loss$��?

accuracy/accuracy_1��2?�)�<       ȷ�R	��Oʑ�A�_*.

	loss/loss���?

accuracy/accuracy_1��4?5��<       ȷ�R	�Rʑ�A�`*.

	loss/loss:��?

accuracy/accuracy_1c�:?��S�<       ȷ�R	xB�Uʑ�A�`*.

	loss/loss��?

accuracy/accuracy_1:9:?�\�<       ȷ�R	�U�Wʑ�A�a*.

	loss/lossn�?

accuracy/accuracy_1�#<?Ba�<       ȷ�R	j�Zʑ�A�b*.

	loss/lossu�?

accuracy/accuracy_1��??4�	 <       ȷ�R	���]ʑ�A�c*.

	loss/loss�ˮ?

accuracy/accuracy_1#�B?'��<       ȷ�R	��`ʑ�A�d*.

	loss/loss�g�?

accuracy/accuracy_1UrA?�CƎ<       ȷ�R	NČcʑ�A�d*.

	loss/loss
�?

accuracy/accuracy_1�F?�J�:<       ȷ�R	���eʑ�A�e*.

	loss/loss�П?

accuracy/accuracy_1��H?�䡺<       ȷ�R	!iʑ�A�f*.

	loss/loss^a�?

accuracy/accuracy_1� I?�A�<       ȷ�R	�=2lʑ�A�g*.

	loss/lossހ�?

accuracy/accuracy_1��J?j�C<       ȷ�R	7joʑ�A�g*.

	loss/loss��?

accuracy/accuracy_1]uM?��Y<       ȷ�R	MVpqʑ�A�h*.

	loss/loss)�?

accuracy/accuracy_1dN?z6g"<       ȷ�R	��\tʑ�A�i*.

	loss/loss8b�?

accuracy/accuracy_1�N?��w_<       ȷ�R	u�qwʑ�A�j*.

	loss/loss"~�?

accuracy/accuracy_1�1O?��+�<       ȷ�R	�)�zʑ�A�k*.

	loss/loss���?

accuracy/accuracy_1� R?F3{2<       ȷ�R	s�}ʑ�A�k*.

	loss/lossx�?

accuracy/accuracy_1.�R?��Ѝ<       ȷ�R	ălʑ�A�l*.

	loss/loss1�}?

accuracy/accuracy_15�S?PJ��<       ȷ�R	R硂ʑ�A�m*.

	loss/lossL�t?

accuracy/accuracy_1�-V?{�]<       ȷ�R	��υʑ�A�n*.

	loss/loss�T{?

accuracy/accuracy_1j�U?tX�3<       ȷ�R	*�	�ʑ�A�n*.

	loss/loss��l?

accuracy/accuracy_1��W?u���<       ȷ�R	��ʑ�A�o*.

	loss/lossy�c?

accuracy/accuracy_1{�X?��<       ȷ�R	���ʑ�A�p*.

	loss/loss�s?

accuracy/accuracy_1g�W?�{C�<       ȷ�R	�q4�ʑ�A�q*.

	loss/lossF�\?

accuracy/accuracy_1�[?���(<       ȷ�R	ycQ�ʑ�A�r*.

	loss/lossl�\?

accuracy/accuracy_1�M\?n0�<       ȷ�R	�&�ʑ�A�r*.

	loss/lossT�e?

accuracy/accuracy_1
[?�JCI<       ȷ�R	:60�ʑ�A�s*.

	loss/loss��S?

accuracy/accuracy_12�\?�X��<       ȷ�R	2�F�ʑ�A�t*.

	loss/lossДM?

accuracy/accuracy_1��]?���<       ȷ�R	��\�ʑ�A�u*.

	loss/lossP�I?

accuracy/accuracy_1��_?��-<       ȷ�R	ƭ��ʑ�A�u*.

	loss/loss'tD?

accuracy/accuracy_1(`?&e�<       ȷ�R	���ʑ�A�v*.

	loss/loss��@?

accuracy/accuracy_1�sa?hI؛<       ȷ�R	w�^�ʑ�A�w*.

	loss/loss��@?

accuracy/accuracy_1��`?���]<       ȷ�R	s9��ʑ�A�x*.

	loss/loss�:?

accuracy/accuracy_1�b?g]<       ȷ�R	B���ʑ�A�y*.

	loss/loss�B?

accuracy/accuracy_1�6`?��Ku<       ȷ�R	�ԡ�ʑ�A�y*.

	loss/lossk�:?

accuracy/accuracy_1[�c?r��?<       ȷ�R	I�ֲʑ�A�z*.

	loss/loss۬3?

accuracy/accuracy_1i/d?�Z`�<       ȷ�R	�0V�ʑ�A�{*.

	loss/loss=�4?

accuracy/accuracy_1��c?9�$<       ȷ�R	��m�ʑ�A�|*.

	loss/loss]K;?

accuracy/accuracy_1��b?r���<       ȷ�R	���ʑ�A�}*.

	loss/lossL�5?

accuracy/accuracy_1�d?Q�em<       ȷ�R	�К�ʑ�A�}*.

	loss/loss��0?

accuracy/accuracy_1�\e?���<       ȷ�R	'��ʑ�A�~*.

	loss/loss�t7?

accuracy/accuracy_1b]d?#��<       ȷ�R	����ʑ�A�*.

	loss/loss�8-?

accuracy/accuracy_1�Re?����=       `I��	?K��ʑ�A��*.

	loss/loss��*?

accuracy/accuracy_15�f?_T�=       `I��	���ʑ�A�*.

	loss/loss�&?

accuracy/accuracy_1��f?p�0�=       `I��	A���ʑ�A؁*.

	loss/lossp�?

accuracy/accuracy_1��g?���U=       `I��	����ʑ�A��*.

	loss/loss%-.?

accuracy/accuracy_1�zf?b�V�=       `I��	0���ʑ�A��*.

	loss/loss�� ?

accuracy/accuracy_13"i?�F=       `I��	֭��ʑ�A��*.

	loss/loss�?

accuracy/accuracy_1cZi?����=       `I��	���ʑ�A�*.

	loss/loss:1%?

accuracy/accuracy_1l�g?6�ǘ=       `I��	���ʑ�A̅*.

	loss/lossK#?

accuracy/accuracy_1Çj?	�=       `I��	����ʑ�A��*.

	loss/loss[ ?

accuracy/accuracy_1A�h?�m�=       `I��	L��ʑ�A��*.

	loss/lossX�?

accuracy/accuracy_1�Dk?GE�=       `I��	����ʑ�A��*.

	loss/loss�?

accuracy/accuracy_1>�k??eJ=       `I��	L��ʑ�A܈*.

	loss/lossRc?

accuracy/accuracy_1�Xl?����=       `I��	 ���ʑ�A��*.

	loss/lossg?

accuracy/accuracy_1M^k?d�D=       `I��	ʗ��ʑ�A��*.

	loss/loss��?

accuracy/accuracy_1g l?�''(=       `I��	�o�ʑ�A��*.

	loss/loss�^?

accuracy/accuracy_1�mk?' ��=       `I��	�YG�ʑ�A�*.

	loss/loss5<?

accuracy/accuracy_1��l?���=       `I��	�A�ʑ�AЌ*.

	loss/lossE�?

accuracy/accuracy_1f�m?��z�=       `I��	QWH�ʑ�A��*.

	loss/loss��?

accuracy/accuracy_1K�l?g,Nj=       `I��	{�\�ʑ�A��*.

	loss/loss��?

accuracy/accuracy_1��l??YC�=       `I��	���ʑ�A��*.

	loss/lossF?

accuracy/accuracy_1��m?W�?k=       `I��	V�4�ʑ�A��*.

	loss/lossP�
?

accuracy/accuracy_1�m?
�T�=       `I��	�/7ˑ�AĐ*.

	loss/loss�y?

accuracy/accuracy_1�4l?�%��=       `I��	<�Gˑ�A��*.

	loss/lossBB?

accuracy/accuracy_1��m?��M=       `I��	�wˑ�A��*.

	loss/loss�Z?

accuracy/accuracy_1f�m?��9=       `I��	�?�
ˑ�A�*.

	loss/lossW?

accuracy/accuracy_1(�m?A�\d=       `I��	��pˑ�Aԓ*.

	loss/loss)<?

accuracy/accuracy_1�n?i��=       `I��	��ˑ�A��*.

	loss/loss�\?

accuracy/accuracy_1�an?�?�
=       `I��	��ˑ�A��*.

	loss/loss�/?

accuracy/accuracy_1ծm?a@�=       `I��	`�ˑ�A��*.

	loss/loss�Q?

accuracy/accuracy_1t>m?&�Z=       `I��	K��ˑ�A�*.

	loss/loss�[?

accuracy/accuracy_1��m?�.|=       `I��	��}ˑ�Aȗ*.

	loss/loss|h�>

accuracy/accuracy_1N[p?5}�\=       `I��	U��ˑ�A��*.

	loss/loss��?

accuracy/accuracy_1��o?����=       `I��	B�C!ˑ�A��*.

	loss/loss�]
?

accuracy/accuracy_1�Lo?����=       `I��	��<#ˑ�A��*.

	loss/lossK=?

accuracy/accuracy_1�Ap?h=       `I��	�8t%ˑ�Aؚ*.

	loss/loss�0?

accuracy/accuracy_1
3o?��=       `I��	6��'ˑ�A��*.

	loss/loss��?

accuracy/accuracy_1��o?[S�=       `I��	Z-9*ˑ�A��*.

	loss/lossc�?

accuracy/accuracy_1��o? ��=       `I��	�Ͷ,ˑ�A��*.

	loss/loss�# ?

accuracy/accuracy_1��p?k�s<=       `I��	�R�.ˑ�A�*.

	loss/loss�=?

accuracy/accuracy_1^�n?ѭ�,=       `I��	�Z1ˑ�A̞*.

	loss/loss�5?

accuracy/accuracy_1�#o?4ș=       `I��	E��3ˑ�A��*.

	loss/loss�c?

accuracy/accuracy_1ʍq?/|�A=       `I��	�/#6ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1ûq?����=       `I��	���8ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1��q?�	%�=       `I��	pg�;ˑ�Aܡ*.

	loss/loss5_�>

accuracy/accuracy_1��q?S\�<=       `I��	m]�=ˑ�A��*.

	loss/loss��?

accuracy/accuracy_1Nq?~�w=       `I��	o�J@ˑ�A��*.

	loss/loss�L�>

accuracy/accuracy_1��p?,�g�=       `I��	���Bˑ�A��*.

	loss/loss�?

accuracy/accuracy_1�Zq?J���=       `I��	���Dˑ�A�*.

	loss/lossyP?

accuracy/accuracy_1iq?��c=       `I��	�9�Fˑ�AХ*.

	loss/loss�<�>

accuracy/accuracy_1�r?��4\=       `I��	�gZIˑ�A��*.

	loss/loss��?

accuracy/accuracy_1d�o?��.5=       `I��	+��Kˑ�A��*.

	loss/loss���>

accuracy/accuracy_1ʍq?���'=       `I��	�xMˑ�A��*.

	loss/loss���>

accuracy/accuracy_1��r?v�%�=       `I��	���Oˑ�A�*.

	loss/lossyv�>

accuracy/accuracy_1R!s?	S�b=       `I��	�L�Qˑ�Aĩ*.

	loss/loss���>

accuracy/accuracy_1Y�s?�6Z=       `I��	�Tˑ�A��*.

	loss/lossŹ�>

accuracy/accuracy_1֫r?��
�=       `I��	�/Vˑ�A��*.

	loss/loss�?

accuracy/accuracy_1p�p?��c=       `I��	Of�Wˑ�A�*.

	loss/lossӫ�>

accuracy/accuracy_1|�s?鸎=       `I��	�I0Zˑ�AԬ*.

	loss/losseK�>

accuracy/accuracy_1�s?ύw=       `I��	Kf\ˑ�A��*.

	loss/lossE��>

accuracy/accuracy_1�%t?c�#�=       `I��	fu�^ˑ�A��*.

	loss/lossg0�>

accuracy/accuracy_17s?��D�=       `I��	x`ˑ�A��*.

	loss/loss�a�>

accuracy/accuracy_1|�r?h7�=       `I��	r	�bˑ�A�*.

	loss/lossE8�>

accuracy/accuracy_1�s?��:}=       `I��	en�dˑ�AȰ*.

	loss/loss��>

accuracy/accuracy_1s?��#=       `I��	�Zgˑ�A��*.

	loss/loss���>

accuracy/accuracy_1��s?1xo=       `I��	bJiiˑ�A��*.

	loss/loss��>

accuracy/accuracy_1^t?���=       `I��	v+Rkˑ�A��*.

	loss/loss��>

accuracy/accuracy_1�ms?�ʎ=       `I��	���mˑ�Aس*.

	loss/loss���>

accuracy/accuracy_1��s?Dg[�=       `I��	��pˑ�A��*.

	loss/lossX��>

accuracy/accuracy_1D}s?Vǎw=       `I��	j��sˑ�A��*.

	loss/loss�}�>

accuracy/accuracy_1�|t?$��4=       `I��	��uˑ�A��*.

	loss/loss�h�>

accuracy/accuracy_1�t?��=       `I��	[%2xˑ�A�*.

	loss/lossZ��>

accuracy/accuracy_1)xs?xWkv=       `I��	Qv�zˑ�A̷*.

	loss/lossp�>

accuracy/accuracy_1!ct?K�y=       `I��	
��}ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1�It?e1Q=       `I��	F{P�ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�t?���K=       `I��	�]H�ˑ�A��*.

	loss/loss�@�>

accuracy/accuracy_1�s?���F=       `I��	aq�ˑ�Aܺ*.

	loss/lossz��>

accuracy/accuracy_1�+s?h��=       `I��	U��ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�+s?��j�=       `I��	�=��ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1��t?~sx=       `I��	+�ƌˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�hs?8U��=       `I��	Z�0�ˑ�A�*.

	loss/loss�N�>

accuracy/accuracy_1zu?fd�=       `I��	�
��ˑ�Aо*.

	loss/lossf��>

accuracy/accuracy_1e�u?gP%%=       `I��	�״�ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1l]u?v$��=       `I��	��R�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1�$v?�BBC=       `I��	�-B�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1V�u?j�=       `I��	��ˑ�A��*.

	loss/loss�K�>

accuracy/accuracy_1x{v?�2=       `I��	a�ʞˑ�A��*.

	loss/loss��>

accuracy/accuracy_1�Mv?���=       `I��	���ˑ�A��*.

	loss/lossW��>

accuracy/accuracy_1�Mv?U��O=       `I��	��ˑ�A��*.

	loss/lossHa�>

accuracy/accuracy_1�av?���=       `I��	�r�ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�u?-�=       `I��	;$5�ˑ�A��*.

	loss/loss"��>

accuracy/accuracy_1��u?���	=       `I��	\���ˑ�A��*.

	loss/loss�>

accuracy/accuracy_1�Rv?Rv�C=       `I��	cˑ�A��*.

	loss/loss:J�>

accuracy/accuracy_1kv?<���=       `I��	��ˑ�A��*.

	loss/loss(R�>

accuracy/accuracy_1�t?���=       `I��	�Fq�ˑ�A��*.

	loss/loss#?

accuracy/accuracy_1��p?8��=       `I��	n6��ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1<%u?��v	=       `I��	���ˑ�A��*.

	loss/lossI3�>

accuracy/accuracy_1�qu?V�O;=       `I��	i��ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�v?���&=       `I��	\�%�ˑ�A��*.

	loss/loss'C�>

accuracy/accuracy_1�Rv?6>	=       `I��	b��ˑ�A��*.

	loss/loss�c�>

accuracy/accuracy_1cw?��!=       `I��	�n��ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1��v?��w=       `I��	ȕ�ˑ�A��*.

	loss/loss�/�>

accuracy/accuracy_1�v?��c=       `I��	��ˑ�A��*.

	loss/loss �>

accuracy/accuracy_1cw?5�B�=       `I��	�]a�ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1H w?R� |=       `I��	���ˑ�A��*.

	loss/lossf��>

accuracy/accuracy_1��v?5�ۦ=       `I��	\Bq�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1\3w?(3Q=       `I��	?΢�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1Aqv?�|��=       `I��	�w��ˑ�A��*.

	loss/lossM�>

accuracy/accuracy_1ĸv?S�U5=       `I��	���ˑ�A��*.

	loss/loss62�>

accuracy/accuracy_1�v?�0��=       `I��		�O�ˑ�A��*.

	loss/loss}��>

accuracy/accuracy_1&lv?･�=       `I��	Z8��ˑ�A��*.

	loss/loss�n�>

accuracy/accuracy_1V�v?8d=       `I��	�y��ˑ�A��*.

	loss/loss��?

accuracy/accuracy_1=�s?ٶ�=       `I��	_�~�ˑ�A��*.

	loss/lossS�?

accuracy/accuracy_1�tp?��v*=       `I��	��3�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1^t?�̀�=       `I��	���ˑ�A��*.

	loss/loss�C�>

accuracy/accuracy_1^�u?d1�=       `I��	�� �ˑ�A��*.

	loss/loss�n�>

accuracy/accuracy_1�u?v�;�=       `I��	8�f�ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1��v?U��=       `I��	� �ˑ�A��*.

	loss/loss� �>

accuracy/accuracy_1�av?Ne�=       `I��	� ��ˑ�A��*.

	loss/loss��>

accuracy/accuracy_1�v?��5g=       `I��	{��ˑ�A��*.

	loss/loss?�>

accuracy/accuracy_1�w?��8=       `I��	��`�ˑ�A��*.

	loss/lossg.�>

accuracy/accuracy_1ˊv?��A�=       `I��	���ˑ�A��*.

	loss/loss ��>

accuracy/accuracy_1@.w?�*�=       `I��	����ˑ�A��*.

	loss/loss�j�>

accuracy/accuracy_1
w?i`e�=       `I��	���ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1�w?M�=       `I��	����ˑ�A��*.

	loss/loss�c�>

accuracy/accuracy_1x8w?��w=       `I��	
�ˑ�A��*.

	loss/loss���>

accuracy/accuracy_1x8w?�ǎ�=       `I��	9�̑�A��*.

	loss/lossK��>

accuracy/accuracy_1	$w?}�'=       `I��	Il̑�A��*.

	loss/loss�w�>

accuracy/accuracy_1Rw?h��=       `I��	כw̑�A��*.

	loss/loss���>

accuracy/accuracy_1,�v?.VI=       `I��	�U̑�A��*.

	loss/loss���>

accuracy/accuracy_1\3w?iU��=       `I��	���
̑�A��*.

	loss/loss��>

accuracy/accuracy_1�w?ǊV�=       `I��	�0̑�A��*.

	loss/lossz��>

accuracy/accuracy_1\3w?)�S�=       `I��	�;x̑�A��*.

	loss/lossBj�>

accuracy/accuracy_1\3w?l�Z=       `I��	W��̑�A��*.

	loss/loss�A�>

accuracy/accuracy_1�uw?��=       `I��	�?�̑�A��*.

	loss/loss3?

accuracy/accuracy_1�r?f8��=       `I��	�k̑�A��*.

	loss/loss���>

accuracy/accuracy_1-�u?%��0=       `I��	W�p̑�A��*.

	loss/loss���>

accuracy/accuracy_1kv?����=       `I��	���̑�A��*.

	loss/loss;��>

accuracy/accuracy_1]vv?AE,m=       `I��	=��̑�A��*.

	loss/loss���>

accuracy/accuracy_1�w?�7u=       `I��	�t�̑�A��*.

	loss/losse��>

accuracy/accuracy_1�v?��_�=       `I��	G<>"̑�A��*.

	loss/loss���>

accuracy/accuracy_1Rw?�x�=       `I��	fQ%̑�A��*.

	loss/loss���>

accuracy/accuracy_1\3w?�ʹ=       `I��	��G'̑�A��*.

	loss/loss	s�>

accuracy/accuracy_1@.w?��f�=       `I��	�
�)̑�A��*.

	loss/loss�_�>

accuracy/accuracy_1�=w?�b�=       `I��	xE,̑�A��*.

	loss/loss��>

accuracy/accuracy_1�Lw?6X��=       `I��	�/̑�A��*.

	loss/lossl}�>

accuracy/accuracy_1�Bw?UO"�=       `I��	1�1̑�A��*.

	loss/loss� �>

accuracy/accuracy_19\w?y�d=       `I��	iя3̑�A��*.

	loss/lossי�>

accuracy/accuracy_1\3w?��0�=       `I��	�A6̑�A��*.

	loss/loss_�>

accuracy/accuracy_1�w?%E3�=       `I��	��
9̑�A��*.

	loss/loss8g�>

accuracy/accuracy_1�uw?��{=       `I��	(�;̑�A��*.

	loss/loss�T�>

accuracy/accuracy_1Uaw?E#��=       `I��	��>̑�A��*.

	loss/loss�%�>

accuracy/accuracy_1بw?��#=       `I��		:U@̑�A��*.

	loss/loss,��>

accuracy/accuracy_1Ww?$�y�=       `I��	��C̑�A��*.

	loss/lossӦ�>

accuracy/accuracy_1N�w?>=       `I��	���Ȇ�A��*.

	loss/loss^;�>

accuracy/accuracy_19\w?� �>=       `I��	O̔H̑�A��*.

	loss/loss���>

accuracy/accuracy_1qfw?��*�=       `I��	MpJ̑�A��*.

	loss/loss�q�>

accuracy/accuracy_1qfw?d�=       `I��	 M̑�A��*.

	loss/lossm�>

accuracy/accuracy_1�w?F���=       `I��	�#�Ȏ�A��*.

	loss/loss�_�>

accuracy/accuracy_1�w??���=       `I��	@.�Ȓ�A��*.

	loss/loss>r�>

accuracy/accuracy_1H w?;�I=       `I��	��Ȗ�A��*.

	loss/lossb=�>

accuracy/accuracy_1R�s?/��$=       `I��	�{HW̑�A��*.

	loss/loss��>

accuracy/accuracy_1�v?t��=       `I��	�)Z̑�A��*.

	loss/loss���>

accuracy/accuracy_1ĸv?k��=       `I��	���\̑�A��*.

	loss/lossM��>

accuracy/accuracy_1Uaw?Y�I=       `I��	:o�_̑�A��*.

	loss/loss���>

accuracy/accuracy_1@.w?� DK=       `I��	��iȃ�A��*.

	loss/loss�Z�>

accuracy/accuracy_1@.w?�Y��=       `I��	��d̑�A��*.

	loss/loss�6�>

accuracy/accuracy_1qfw?��m�=       `I��	�f�f̑�A�*.

	loss/loss���>

accuracy/accuracy_1b�w?�䝵=       `I��	���ȋ�A̂*.

	loss/lossq%�>

accuracy/accuracy_1��w?%vd�=       `I��	�\�k̑�A��*.

	loss/loss���>

accuracy/accuracy_1b�w?:��=       `I��	$sn̑�A��*.

	loss/loss6S�>

accuracy/accuracy_1w�w?�B��=       `I��	[��p̑�A��*.

	loss/loss��>

accuracy/accuracy_1بw?;RDq=       `I��	Еs̑�A܅*.

	loss/loss3��>

accuracy/accuracy_1+�w?����=       `I��	�Uv̑�A��*.

	loss/loss�k�>

accuracy/accuracy_1�kw?~vA�=       `I��	/�9x̑�A��*.

	loss/loss�k�>

accuracy/accuracy_1�w?!��=       `I��	4\�z̑�A��*.

	loss/loss�t�>

accuracy/accuracy_1�w?1Ok=       `I��	��}̑�A�*.

	loss/loss���>

accuracy/accuracy_1��w?.nz%=       `I��	��e�̑�AЉ*.

	loss/loss'��>

accuracy/accuracy_1N�w?�|==       `I��	�Â̑�A��*.

	loss/lossR|�>

accuracy/accuracy_1b�w?Y���=       `I��	���̑�A��*.

	loss/loss�=�>

accuracy/accuracy_1�	x?��8�=       `I��	ٛ��̑�A��*.

	loss/loss�}�>

accuracy/accuracy_1�uw?-���=       `I��	�邊̑�A��*.

	loss/loss
��>

accuracy/accuracy_1j�w?d��/=       `I��	�B�̑�Ač*.

	loss/loss�t�>

accuracy/accuracy_19\w?q���=       `I��	��+�̑�A��*.

	loss/lossA��>

accuracy/accuracy_1,�v?�K��=       `I��	Ȏ̑�A��*.

	loss/loss�+?

accuracy/accuracy_1eWn?'���=       `I��	����̑�A��*.

	loss/lossb��>

accuracy/accuracy_1�Hu?gE��=       `I��	�Q�̑�AԐ*.

	loss/loss��>

accuracy/accuracy_1�w?1{ �=       `I��	
®�̑�A��*.

	loss/loss
s�>

accuracy/accuracy_1�w?Y��a=       `I��	 ��̑�A��*.

	loss/loss���>

accuracy/accuracy_1->v?u�>J=       `I��	!E֞̑�A��*.

	loss/loss�M�>

accuracy/accuracy_1j�w?��;�=       `I��	�ƛ�̑�A�*.

	loss/loss��>

accuracy/accuracy_1�kw?���a=       `I��	��d�̑�AȔ*.

	loss/loss��>

accuracy/accuracy_1��w?:�VQ=       `I��	�jR�̑�A��*.

	loss/loss�(�>

accuracy/accuracy_1�=w?_~C=       `I��	�2�̑�A��*.

	loss/loss���>

accuracy/accuracy_1ĸv?���n=       `I��	%z��̑�A��*.

	loss/loss�߾>

accuracy/accuracy_1w�w?��L=       `I��	��{�̑�Aؗ*.

	loss/lossA9�>

accuracy/accuracy_1j�w?�]6�=       `I��	���̑�A��*.

	loss/loss��>

accuracy/accuracy_1�\v?�<��=       `I��	%�
�̑�A��*.

	loss/lossP��>

accuracy/accuracy_1�vu?X�k=       `I��	\е̑�A��*.

	loss/loss���>

accuracy/accuracy_1��v?__ʎ=       `I��	�ʗ�̑�A�*.

	loss/loss�~�>

accuracy/accuracy_1�w?v��m=       `I��	w�T�̑�A̛*.

	loss/lossu��>

accuracy/accuracy_1Rw?��c�=       `I��	��H�̑�A��*.

	loss/lossb�>

accuracy/accuracy_1	$w?�qd
=       `I��	��ؿ̑�A��*.

	loss/loss���>

accuracy/accuracy_1�w?��{�=       `I��	����̑�A��*.

	loss/loss,�>

accuracy/accuracy_1�kw?�P~=       `I��	�Y�̑�Aܞ*.

	loss/lossuA�>

accuracy/accuracy_1��w?���)=       `I��	ݶ��̑�A��*.

	loss/loss���>

accuracy/accuracy_1w�w?*#)\=       `I��	�/��̑�A��*.

	loss/lossV��>

accuracy/accuracy_1��w?�ٌ=       `I��	����̑�A��*.

	loss/lossx��>

accuracy/accuracy_1�kw?�Za=       `I��	[N_�̑�A�*.

	loss/loss|m�>

accuracy/accuracy_1\3w?4�τ=       `I��	M�̑�AТ*.

	loss/loss|��>

accuracy/accuracy_1�>u?&0�=       `I��	�}�̑�A��*.

	loss/lossȟ�>

accuracy/accuracy_1�pw?���=       `I��	���̑�A��*.

	loss/lossƵ�>

accuracy/accuracy_1
w?"�a�=       `I��	�~f�̑�A��*.

	loss/loss�t�>

accuracy/accuracy_1MLx?�ʙ�=       `I��	u-�̑�A�*.

	loss/loss?��>

accuracy/accuracy_1�7x?:�'=       `I��	^'��̑�AĦ*.

	loss/loss�)�>

accuracy/accuracy_1p#x?�b�=       `I��	\V��̑�A��*.

	loss/loss��>

accuracy/accuracy_1�7x?��s=       `I��	�6��̑�A��*.

	loss/loss��>

accuracy/accuracy_1��x?Kqm�=       `I��	��F�̑�A�*.

	loss/lossպ�>

accuracy/accuracy_1Tx?GP7k=       `I��	<�̑�Aԩ*.

	loss/loss���>

accuracy/accuracy_1�-x?�л�=       `I��	o��̑�A��*.

	loss/loss��>

accuracy/accuracy_1p#x?W�J=       `I��	$+��̑�A��*.

	loss/loss���>

accuracy/accuracy_1bx?�'�&=       `I��	�'h�̑�A��*.

	loss/loss��>

accuracy/accuracy_1x?�P=       `I��	��*�̑�A�*.

	loss/loss���>

accuracy/accuracy_1px?� ��=       `I��	����̑�Aȭ*.

	loss/losso��>

accuracy/accuracy_1Bx?�4%�=       `I��	毶�̑�A��*.

	loss/loss.=�>

accuracy/accuracy_1�w?^��=       `I��	Uz�̑�A��*.

	loss/lossU�>

accuracy/accuracy_1�zw?;9=       `I��	�FC�̑�A��*.

	loss/loss���>

accuracy/accuracy_1��t?Ȝ��=       `I��	? ͑�Aذ*.

	loss/loss�b�>

accuracy/accuracy_1�w?�;��=       `I��	�p͑�A��*.

	loss/loss_)�>

accuracy/accuracy_1��w?C/=       `I��	?Q�͑�A��*.

	loss/lossſ�>

accuracy/accuracy_1�x?x*=       `I��	�"\͑�A��*.

	loss/loss+��>

accuracy/accuracy_1MLx?�+��=       `I��	�u!
͑�A�*.

	loss/loss��>

accuracy/accuracy_1Гx?�ȳ=       `I��	��͑�A̴*.

	loss/loss�*�>

accuracy/accuracy_1��x?��Hh=       `I��	���͑�A��*.

	loss/lossi��>

accuracy/accuracy_1Гx?U6��=       `I��	�i͑�A��*.

	loss/loss�6�>

accuracy/accuracy_1o�x?�^�=       `I��	2�&͑�A��*.

	loss/loss���>

accuracy/accuracy_1�2x?��y�=       `I��	)c�͑�Aܷ*.

	loss/loss�5�>

accuracy/accuracy_1�ex?f�JD=       `I��	Na�͑�A��*.

	loss/loss���>

accuracy/accuracy_1�x?�c��=       `I��	j(�͑�A��*.

	loss/loss�M�>

accuracy/accuracy_1[�x?��h�=       `I��	��W͑�A��*.

	loss/loss[��>

accuracy/accuracy_1iQx?�ص�=       `I��	�"
!͑�A�*.

	loss/lossRC�>

accuracy/accuracy_1�jx?/=       `I��	-||#͑�Aл*.

	loss/loss���>

accuracy/accuracy_1�x?���=       `I��	 ��%͑�A��*.

	loss/loss�6�>

accuracy/accuracy_1��x?Lo�i=       `I��	Ђ6(͑�A��*.

	loss/loss���>

accuracy/accuracy_1��x?��XM=       `I��	:��*͑�A��*.

	loss/loss���>

accuracy/accuracy_1��w?���=       `I��	+צ-͑�A�*.

	loss/loss�>

accuracy/accuracy_1��x?��Y=       `I��	�'�/͑�AĿ*.

	loss/loss {�>

accuracy/accuracy_1iQx?��C�=       `I��	��&2͑�A��*.

	loss/loss���>

accuracy/accuracy_1v�x?_��?=       `I��	Ν�4͑�A��*.

	loss/loss���>

accuracy/accuracy_1��x?�S�=       `I��	N�7͑�A��*.

	loss/loss�Z�>

accuracy/accuracy_1px?���=       `I��	�/:͑�A��*.

	loss/loss#x�>

accuracy/accuracy_1��w?q%Q�=       `I��	.�4<͑�A��*.

	loss/loss���>

accuracy/accuracy_1.�t?O=�C=       `I��	e��>͑�A��*.

	loss/loss ��>

accuracy/accuracy_1��v?���[=       `I��	Z��A͑�A��*.

	loss/lossc��>

accuracy/accuracy_1��w?� }�=       `I��	��iD͑�A��*.

	loss/lossX��>

accuracy/accuracy_1��x?,j�w=       `I��	��xF͑�A��*.

	loss/loss���>

accuracy/accuracy_1��x?��S"=       `I��	���H͑�A��*.

	loss/loss�޹>

accuracy/accuracy_1��x?_��=       `I��	}ޞK͑�A��*.

	loss/loss�D�>

accuracy/accuracy_1�x?Df<*=       `I��	�]N͑�A��*.

	loss/loss�>

accuracy/accuracy_1��x?�,=       `I��	���P͑�A��*.

	loss/loss2�>

accuracy/accuracy_1-y?�w<w=       `I��	���R͑�A��*.

	loss/loss���>

accuracy/accuracy_1��x?��ը=       `I��	��U͑�A��*.

	loss/lossJi�>

accuracy/accuracy_1�[x?t��!=       `I��	��iX͑�A��*.

	loss/loss��>

accuracy/accuracy_1��x?�v=       `I��	�`'[͑�A��*.

	loss/lossd��>

accuracy/accuracy_1��x?����=       `I��	[;]͑�A��*.

	loss/loss%w�>

accuracy/accuracy_1hy?�I�F=       `I��	`��_͑�A��*.

	loss/losskA�>

accuracy/accuracy_1#�x?���=       `I��	wnb͑�A��*.

	loss/loss$۽>

accuracy/accuracy_1 �x?/Rk�=       `I��	V?)e͑�A��*.

	loss/loss�`�>

accuracy/accuracy_1[�x?pCR=       `I��	ת�g͑�A��*.

	loss/loss��>

accuracy/accuracy_1��x?���=       `I��	�H�i͑�A��*.

	loss/loss��>

accuracy/accuracy_1�Py?(=��=       `I��	�0jl͑�A��*.

	loss/loss~o�>

accuracy/accuracy_1��w?�ޥ�=       `I��	A,#o͑�A��*.

	loss/loss��>

accuracy/accuracy_1v�x?����=       `I��	�*�q͑�A��*.

	loss/loss^��>

accuracy/accuracy_1v�x?=���=       `I��	��s͑�A��*.

	loss/loss9*�>

accuracy/accuracy_1[�x?b8��=       `I��	��Yv͑�A��*.

	loss/lossڱ�>

accuracy/accuracy_1�<x?K��=       `I��	<�y͑�A��*.

	loss/loss���>

accuracy/accuracy_1��x?%�z�=       `I��	�t�{͑�A��*.

	loss/loss���>

accuracy/accuracy_1��w?�xHs=       `I��	xfe~͑�A��*.

	loss/lossf��>

accuracy/accuracy_1��t?>:��=       `I��	z
]�͑�A��*.

	loss/lossu��>

accuracy/accuracy_1�<x?��n�=       `I��	h �͑�A��*.

	loss/loss9>�>

accuracy/accuracy_1�w?���=       `I��	��߅͑�A��*.

	loss/loss5Z�>

accuracy/accuracy_1#�x?���=       `I��	����͑�A��*.

	loss/lossľ>

accuracy/accuracy_1��x?� s�=       `I��	�>��͑�A��*.

	loss/loss���>

accuracy/accuracy_1�"y?�ƕ=       `I��	_��͑�A��*.

	loss/lossp��>

accuracy/accuracy_1 �x?`� z=       `I��	S�ۏ͑�A��*.

	loss/lossm��>

accuracy/accuracy_1��x?|g/�=       `I��	�T��͑�A��*.

	loss/lossI��>

accuracy/accuracy_1bx?����=       `I��	s�+�͑�A��*.

	loss/losstX�>

accuracy/accuracy_1�'y?!t�=       `I��	�= �͑�A��*.

	loss/lossdU�>

accuracy/accuracy_1�x?|��1=       `I��	���͑�A��*.

	loss/loss��>

accuracy/accuracy_1?�x?�HUS=       `I��	���͑�A��*.

	loss/loss��>

accuracy/accuracy_1v�x?��1=       `I��	��x�͑�A��*.

	loss/loss�>

accuracy/accuracy_1L	y?���^=       `I��	d���͑�A��*.

	loss/losso1�>

accuracy/accuracy_1Fzx?WK=       `I��	���͑�A��*.

	loss/loss���>

accuracy/accuracy_14v?��J=       `I��	n\��͑�A��*.

	loss/loss��>

accuracy/accuracy_1��v?y� s=       `I��	2�m�͑�A��*.

	loss/losshW�>

accuracy/accuracy_1x8w?f���=       `I��	���͑�A��*.

	loss/loss���>

accuracy/accuracy_1iQx?W��=       `I��	i$�͑�A��*.

	loss/loss˩�>

accuracy/accuracy_1*ux?lq~�=       `I��	����͑�A��*.

	loss/lossu|�>

accuracy/accuracy_1��x?�-G!=       `I��	l�S�͑�A��*.

	loss/loss�ְ>

accuracy/accuracy_1*2y?���=       `I��	�F
�͑�A��*.

	loss/loss��>

accuracy/accuracy_1>ey?��^=       `I��	0�͑�A��*.

	loss/loss� �>

accuracy/accuracy_1�Ky?t>�=       `I��	)ކ�͑�A��*.

	loss/loss��>

accuracy/accuracy_1E7y?�O84=       `I��	�=�͑�A��*.

	loss/loss�B�>

accuracy/accuracy_1hy?om��=       `I��	�(�͑�A��*.

	loss/lossR�>

accuracy/accuracy_1�y?�X�=       `I��	?���͑�A��*.

	loss/loss"�>

accuracy/accuracy_1ݱy?��t=       `I��	eHm�͑�A��*.

	loss/lossNS�>

accuracy/accuracy_11y?�q�=       `I��	Z��͑�A��*.

	loss/loss�E�>

accuracy/accuracy_1*2y?5��=       `I��	��͑�A��*.

	loss/losso�>

accuracy/accuracy_1*2y?��b�=       `I��	ET��͑�A��*.

	loss/loss��>

accuracy/accuracy_1�Uy?6Tê=       `I��	"y��͑�A��*.

	loss/lossor�>

accuracy/accuracy_1}Ay?��7`=       `I��	G�Q�͑�A��*.

	loss/loss��>

accuracy/accuracy_1a<y?%$��=       `I��	C�͑�A��*.

	loss/loss��>

accuracy/accuracy_1�y?	{��=       `I��	�Ѿ�͑�A��*.

	loss/loss���>

accuracy/accuracy_1�Ky?u�k=       `I��	]xY�͑�A��*.

	loss/loss�Q�>

accuracy/accuracy_1��y?���=       `I��	N�<�͑�A��*.

	loss/loss崾>

accuracy/accuracy_1o�x?GF��=       `I��	C���͑�A��*.

	loss/lossL��>

accuracy/accuracy_1%)w?�,�V=       `I��	[{��͑�A��*.

	loss/loss��>

accuracy/accuracy_1G�w?��@=       `I��	�dj�͑�A��*.

	loss/losskS�>

accuracy/accuracy_18�x?7��=       `I��	���͑�A��*.

	loss/lossޫ�>

accuracy/accuracy_1�y?#T�=       `I��	T���͑�A��*.

	loss/loss���>

accuracy/accuracy_1�y?'��=       `I��	�ˢ�͑�A��*.

	loss/loss�?�>

accuracy/accuracy_1��y?q��<=       `I��	Q�e�͑�A��*.

	loss/loss�~�>

accuracy/accuracy_1g�y?v�p�=       `I��	G��͑�A��*.

	loss/loss�̮>

accuracy/accuracy_1�ty?Q��(=       `I��	)���͑�A��*.

	loss/lossC��>

accuracy/accuracy_1�y?���=       `I��	Z���͑�A��*.

	loss/loss享>

accuracy/accuracy_1��y?��=       `I��	��`�͑�A��*.

	loss/lossD�>

accuracy/accuracy_1�y?���;=       `I��	���͑�A��*.

	loss/loss�-�>

accuracy/accuracy_1��y? ���=       `I��	=kM�͑�A��*.

	loss/loss���>

accuracy/accuracy_1Zjy?9��"=       `I��	�!��͑�A��*.

	loss/loss��>

accuracy/accuracy_1S�y?���=       `I��	l'VΑ�A��*.

	loss/lossUH�>

accuracy/accuracy_1��y?�tv =       `I��	!
Α�A��*.

	loss/loss��>

accuracy/accuracy_1��y?�`�z=       `I��	`ӯΑ�A��*.

	loss/lossk%�>

accuracy/accuracy_1�Uy?}u�=       `I��	��Α�A܂*.

	loss/lossf��>

accuracy/accuracy_1E7y?n'=       `I��	e:<Α�A��*.

	loss/loss ^�>

accuracy/accuracy_1�y?@�$H=       `I��	���Α�A��*.

	loss/loss�U3?

accuracy/accuracy_1$�p?k��#=       `I��	ةΑ�A��*.

	loss/loss.�>

accuracy/accuracy_1Bx?r�:=       `I��	���Α�A�*.

	loss/loss�c�>

accuracy/accuracy_1�y?
�]=       `I��	WO'Α�AІ*.

	loss/loss�7�>

accuracy/accuracy_1�~y?L��!