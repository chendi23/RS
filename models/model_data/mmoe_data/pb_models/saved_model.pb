??
?.?.
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?

ScatterAdd
ref"T?
indices"Tindices
updates"T

output_ref"T?" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
?
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.14.02unknown??
i
u_typePlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
h
u_agePlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
h
u_sexPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
k
u_pos_idPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
l
	u_seat_idPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
k
u_org_idPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
p
i_class_labelPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
m

i_entitiesPlaceholder*
shape:?????????@*
dtype0*'
_output_shapes
:?????????@
n
PlaceholderPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
p
Placeholder_1Placeholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
t
#user_embedding/random_uniform/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
f
!user_embedding/random_uniform/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
f
!user_embedding/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
+user_embedding/random_uniform/RandomUniformRandomUniform#user_embedding/random_uniform/shape*
seed?*
T0*
dtype0*
seed2*
_output_shapes

:

?
!user_embedding/random_uniform/subSub!user_embedding/random_uniform/max!user_embedding/random_uniform/min*
T0*
_output_shapes
: 
?
!user_embedding/random_uniform/mulMul+user_embedding/random_uniform/RandomUniform!user_embedding/random_uniform/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniformAdd!user_embedding/random_uniform/mul!user_embedding/random_uniform/min*
T0*
_output_shapes

:

?
 user_embedding/u_type_emb_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
'user_embedding/u_type_emb_matrix/AssignAssign user_embedding/u_type_emb_matrixuser_embedding/random_uniform*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
%user_embedding/u_type_emb_matrix/readIdentity user_embedding/u_type_emb_matrix*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
$user_embedding/u_type_emb_layer/axisConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_type_emb_layerGatherV2%user_embedding/u_type_emb_matrix/readu_type$user_embedding/u_type_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*+
_output_shapes
:?????????

?
(user_embedding/u_type_emb_layer/IdentityIdentityuser_embedding/u_type_emb_layer*
T0*+
_output_shapes
:?????????

v
%user_embedding/random_uniform_1/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
h
#user_embedding/random_uniform_1/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
#user_embedding/random_uniform_1/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-user_embedding/random_uniform_1/RandomUniformRandomUniform%user_embedding/random_uniform_1/shape*
seed?*
T0*
dtype0*
seed2*
_output_shapes

:

?
#user_embedding/random_uniform_1/subSub#user_embedding/random_uniform_1/max#user_embedding/random_uniform_1/min*
T0*
_output_shapes
: 
?
#user_embedding/random_uniform_1/mulMul-user_embedding/random_uniform_1/RandomUniform#user_embedding/random_uniform_1/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniform_1Add#user_embedding/random_uniform_1/mul#user_embedding/random_uniform_1/min*
T0*
_output_shapes

:

?
user_embedding/u_age_emn_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
&user_embedding/u_age_emn_matrix/AssignAssignuser_embedding/u_age_emn_matrixuser_embedding/random_uniform_1*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
$user_embedding/u_age_emn_matrix/readIdentityuser_embedding/u_age_emn_matrix*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
#user_embedding/u_age_emb_layer/axisConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_age_emb_layerGatherV2$user_embedding/u_age_emn_matrix/readu_age#user_embedding/u_age_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*+
_output_shapes
:?????????

?
'user_embedding/u_age_emb_layer/IdentityIdentityuser_embedding/u_age_emb_layer*
T0*+
_output_shapes
:?????????

v
%user_embedding/random_uniform_2/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
h
#user_embedding/random_uniform_2/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
#user_embedding/random_uniform_2/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-user_embedding/random_uniform_2/RandomUniformRandomUniform%user_embedding/random_uniform_2/shape*
seed?*
T0*
dtype0*
seed2'*
_output_shapes

:

?
#user_embedding/random_uniform_2/subSub#user_embedding/random_uniform_2/max#user_embedding/random_uniform_2/min*
T0*
_output_shapes
: 
?
#user_embedding/random_uniform_2/mulMul-user_embedding/random_uniform_2/RandomUniform#user_embedding/random_uniform_2/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniform_2Add#user_embedding/random_uniform_2/mul#user_embedding/random_uniform_2/min*
T0*
_output_shapes

:

?
user_embedding/u_sex_emb_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
&user_embedding/u_sex_emb_matrix/AssignAssignuser_embedding/u_sex_emb_matrixuser_embedding/random_uniform_2*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
$user_embedding/u_sex_emb_matrix/readIdentityuser_embedding/u_sex_emb_matrix*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
#user_embedding/u_sex_emb_layer/axisConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_sex_emb_layerGatherV2$user_embedding/u_sex_emb_matrix/readu_sex#user_embedding/u_sex_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*+
_output_shapes
:?????????

?
'user_embedding/u_sex_emb_layer/IdentityIdentityuser_embedding/u_sex_emb_layer*
T0*+
_output_shapes
:?????????

v
%user_embedding/random_uniform_3/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
h
#user_embedding/random_uniform_3/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
#user_embedding/random_uniform_3/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-user_embedding/random_uniform_3/RandomUniformRandomUniform%user_embedding/random_uniform_3/shape*
seed?*
T0*
dtype0*
seed24*
_output_shapes

:

?
#user_embedding/random_uniform_3/subSub#user_embedding/random_uniform_3/max#user_embedding/random_uniform_3/min*
T0*
_output_shapes
: 
?
#user_embedding/random_uniform_3/mulMul-user_embedding/random_uniform_3/RandomUniform#user_embedding/random_uniform_3/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniform_3Add#user_embedding/random_uniform_3/mul#user_embedding/random_uniform_3/min*
T0*
_output_shapes

:

?
user_embedding/u_org_emb_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
&user_embedding/u_org_emb_matrix/AssignAssignuser_embedding/u_org_emb_matrixuser_embedding/random_uniform_3*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
$user_embedding/u_org_emb_matrix/readIdentityuser_embedding/u_org_emb_matrix*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
#user_embedding/u_org_emb_layer/axisConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_org_emb_layerGatherV2$user_embedding/u_org_emb_matrix/readu_org_id#user_embedding/u_org_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*+
_output_shapes
:?????????

?
'user_embedding/u_org_emb_layer/IdentityIdentityuser_embedding/u_org_emb_layer*
T0*+
_output_shapes
:?????????

v
%user_embedding/random_uniform_4/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
h
#user_embedding/random_uniform_4/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
#user_embedding/random_uniform_4/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-user_embedding/random_uniform_4/RandomUniformRandomUniform%user_embedding/random_uniform_4/shape*
seed?*
T0*
dtype0*
seed2A*
_output_shapes

:

?
#user_embedding/random_uniform_4/subSub#user_embedding/random_uniform_4/max#user_embedding/random_uniform_4/min*
T0*
_output_shapes
: 
?
#user_embedding/random_uniform_4/mulMul-user_embedding/random_uniform_4/RandomUniform#user_embedding/random_uniform_4/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniform_4Add#user_embedding/random_uniform_4/mul#user_embedding/random_uniform_4/min*
T0*
_output_shapes

:

?
 user_embedding/u_seat_emb_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
'user_embedding/u_seat_emb_matrix/AssignAssign user_embedding/u_seat_emb_matrixuser_embedding/random_uniform_4*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
%user_embedding/u_seat_emb_matrix/readIdentity user_embedding/u_seat_emb_matrix*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
$user_embedding/u_seat_emb_layer/axisConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_seat_emb_layerGatherV2%user_embedding/u_seat_emb_matrix/read	u_seat_id$user_embedding/u_seat_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*+
_output_shapes
:?????????

?
(user_embedding/u_seat_emb_layer/IdentityIdentityuser_embedding/u_seat_emb_layer*
T0*+
_output_shapes
:?????????

v
%user_embedding/random_uniform_5/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
h
#user_embedding/random_uniform_5/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
h
#user_embedding/random_uniform_5/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-user_embedding/random_uniform_5/RandomUniformRandomUniform%user_embedding/random_uniform_5/shape*
seed?*
T0*
dtype0*
seed2N*
_output_shapes

:

?
#user_embedding/random_uniform_5/subSub#user_embedding/random_uniform_5/max#user_embedding/random_uniform_5/min*
T0*
_output_shapes
: 
?
#user_embedding/random_uniform_5/mulMul-user_embedding/random_uniform_5/RandomUniform#user_embedding/random_uniform_5/sub*
T0*
_output_shapes

:

?
user_embedding/random_uniform_5Add#user_embedding/random_uniform_5/mul#user_embedding/random_uniform_5/min*
T0*
_output_shapes

:

?
user_embedding/u_pos_emb_matrix
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
&user_embedding/u_pos_emb_matrix/AssignAssignuser_embedding/u_pos_emb_matrixuser_embedding/random_uniform_5*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
$user_embedding/u_pos_emb_matrix/readIdentityuser_embedding/u_pos_emb_matrix*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
#user_embedding/u_pos_emb_layer/axisConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
user_embedding/u_pos_emb_layerGatherV2$user_embedding/u_pos_emb_matrix/readu_pos_id#user_embedding/u_pos_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*+
_output_shapes
:?????????

?
'user_embedding/u_pos_emb_layer/IdentityIdentityuser_embedding/u_pos_emb_layer*
T0*+
_output_shapes
:?????????

?
1u_type_fc/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@u_type_fc/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
/u_type_fc/kernel/Initializer/random_uniform/minConst*#
_class
loc:@u_type_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
/u_type_fc/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@u_type_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
9u_type_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform1u_type_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*#
_class
loc:@u_type_fc/kernel*
dtype0*
seed2[*
_output_shapes

:


?
/u_type_fc/kernel/Initializer/random_uniform/subSub/u_type_fc/kernel/Initializer/random_uniform/max/u_type_fc/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes
: 
?
/u_type_fc/kernel/Initializer/random_uniform/mulMul9u_type_fc/kernel/Initializer/random_uniform/RandomUniform/u_type_fc/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes

:


?
+u_type_fc/kernel/Initializer/random_uniformAdd/u_type_fc/kernel/Initializer/random_uniform/mul/u_type_fc/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes

:


?
u_type_fc/kernel
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_type_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_type_fc/kernel/AssignAssignu_type_fc/kernel+u_type_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_type_fc/kernel/readIdentityu_type_fc/kernel*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes

:


?
 u_type_fc/bias/Initializer/zerosConst*!
_class
loc:@u_type_fc/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_type_fc/bias
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_type_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_type_fc/bias/AssignAssignu_type_fc/bias u_type_fc/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

w
u_type_fc/bias/readIdentityu_type_fc/bias*
T0*!
_class
loc:@u_type_fc/bias*
_output_shapes
:

j
 user_fc/u_type_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
q
 user_fc/u_type_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
!user_fc/u_type_fc/Tensordot/ShapeShape(user_embedding/u_type_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
k
)user_fc/u_type_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
$user_fc/u_type_fc/Tensordot/GatherV2GatherV2!user_fc/u_type_fc/Tensordot/Shape user_fc/u_type_fc/Tensordot/free)user_fc/u_type_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
m
+user_fc/u_type_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
&user_fc/u_type_fc/Tensordot/GatherV2_1GatherV2!user_fc/u_type_fc/Tensordot/Shape user_fc/u_type_fc/Tensordot/axes+user_fc/u_type_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
k
!user_fc/u_type_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
 user_fc/u_type_fc/Tensordot/ProdProd$user_fc/u_type_fc/Tensordot/GatherV2!user_fc/u_type_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
m
#user_fc/u_type_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
"user_fc/u_type_fc/Tensordot/Prod_1Prod&user_fc/u_type_fc/Tensordot/GatherV2_1#user_fc/u_type_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'user_fc/u_type_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
"user_fc/u_type_fc/Tensordot/concatConcatV2 user_fc/u_type_fc/Tensordot/free user_fc/u_type_fc/Tensordot/axes'user_fc/u_type_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
!user_fc/u_type_fc/Tensordot/stackPack user_fc/u_type_fc/Tensordot/Prod"user_fc/u_type_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
%user_fc/u_type_fc/Tensordot/transpose	Transpose(user_embedding/u_type_emb_layer/Identity"user_fc/u_type_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
#user_fc/u_type_fc/Tensordot/ReshapeReshape%user_fc/u_type_fc/Tensordot/transpose!user_fc/u_type_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
}
,user_fc/u_type_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
'user_fc/u_type_fc/Tensordot/transpose_1	Transposeu_type_fc/kernel/read,user_fc/u_type_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


|
+user_fc/u_type_fc/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
%user_fc/u_type_fc/Tensordot/Reshape_1Reshape'user_fc/u_type_fc/Tensordot/transpose_1+user_fc/u_type_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
"user_fc/u_type_fc/Tensordot/MatMulMatMul#user_fc/u_type_fc/Tensordot/Reshape%user_fc/u_type_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

m
#user_fc/u_type_fc/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
k
)user_fc/u_type_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
$user_fc/u_type_fc/Tensordot/concat_1ConcatV2$user_fc/u_type_fc/Tensordot/GatherV2#user_fc/u_type_fc/Tensordot/Const_2)user_fc/u_type_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_type_fc/TensordotReshape"user_fc/u_type_fc/Tensordot/MatMul$user_fc/u_type_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_type_fc/BiasAddBiasAdduser_fc/u_type_fc/Tensordotu_type_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

o
user_fc/u_type_fc/ReluReluuser_fc/u_type_fc/BiasAdd*
T0*+
_output_shapes
:?????????

?
0u_age_fc/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@u_age_fc/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
.u_age_fc/kernel/Initializer/random_uniform/minConst*"
_class
loc:@u_age_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
.u_age_fc/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@u_age_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
8u_age_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform0u_age_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*"
_class
loc:@u_age_fc/kernel*
dtype0*
seed2?*
_output_shapes

:


?
.u_age_fc/kernel/Initializer/random_uniform/subSub.u_age_fc/kernel/Initializer/random_uniform/max.u_age_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes
: 
?
.u_age_fc/kernel/Initializer/random_uniform/mulMul8u_age_fc/kernel/Initializer/random_uniform/RandomUniform.u_age_fc/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes

:


?
*u_age_fc/kernel/Initializer/random_uniformAdd.u_age_fc/kernel/Initializer/random_uniform/mul.u_age_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes

:


?
u_age_fc/kernel
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_age_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_age_fc/kernel/AssignAssignu_age_fc/kernel*u_age_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


~
u_age_fc/kernel/readIdentityu_age_fc/kernel*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes

:


?
u_age_fc/bias/Initializer/zerosConst* 
_class
loc:@u_age_fc/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_age_fc/bias
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_age_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_age_fc/bias/AssignAssignu_age_fc/biasu_age_fc/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

t
u_age_fc/bias/readIdentityu_age_fc/bias*
T0* 
_class
loc:@u_age_fc/bias*
_output_shapes
:

i
user_fc/u_age_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
p
user_fc/u_age_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
 user_fc/u_age_fc/Tensordot/ShapeShape'user_embedding/u_age_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
j
(user_fc/u_age_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_age_fc/Tensordot/GatherV2GatherV2 user_fc/u_age_fc/Tensordot/Shapeuser_fc/u_age_fc/Tensordot/free(user_fc/u_age_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
l
*user_fc/u_age_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
%user_fc/u_age_fc/Tensordot/GatherV2_1GatherV2 user_fc/u_age_fc/Tensordot/Shapeuser_fc/u_age_fc/Tensordot/axes*user_fc/u_age_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
j
 user_fc/u_age_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
user_fc/u_age_fc/Tensordot/ProdProd#user_fc/u_age_fc/Tensordot/GatherV2 user_fc/u_age_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
l
"user_fc/u_age_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
!user_fc/u_age_fc/Tensordot/Prod_1Prod%user_fc/u_age_fc/Tensordot/GatherV2_1"user_fc/u_age_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
&user_fc/u_age_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
!user_fc/u_age_fc/Tensordot/concatConcatV2user_fc/u_age_fc/Tensordot/freeuser_fc/u_age_fc/Tensordot/axes&user_fc/u_age_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
 user_fc/u_age_fc/Tensordot/stackPackuser_fc/u_age_fc/Tensordot/Prod!user_fc/u_age_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
$user_fc/u_age_fc/Tensordot/transpose	Transpose'user_embedding/u_age_emb_layer/Identity!user_fc/u_age_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
"user_fc/u_age_fc/Tensordot/ReshapeReshape$user_fc/u_age_fc/Tensordot/transpose user_fc/u_age_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
|
+user_fc/u_age_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
&user_fc/u_age_fc/Tensordot/transpose_1	Transposeu_age_fc/kernel/read+user_fc/u_age_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


{
*user_fc/u_age_fc/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
$user_fc/u_age_fc/Tensordot/Reshape_1Reshape&user_fc/u_age_fc/Tensordot/transpose_1*user_fc/u_age_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
!user_fc/u_age_fc/Tensordot/MatMulMatMul"user_fc/u_age_fc/Tensordot/Reshape$user_fc/u_age_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

l
"user_fc/u_age_fc/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
j
(user_fc/u_age_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_age_fc/Tensordot/concat_1ConcatV2#user_fc/u_age_fc/Tensordot/GatherV2"user_fc/u_age_fc/Tensordot/Const_2(user_fc/u_age_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_age_fc/TensordotReshape!user_fc/u_age_fc/Tensordot/MatMul#user_fc/u_age_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_age_fc/BiasAddBiasAdduser_fc/u_age_fc/Tensordotu_age_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

m
user_fc/u_age_fc/ReluReluuser_fc/u_age_fc/BiasAdd*
T0*+
_output_shapes
:?????????

?
0u_sex_fc/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@u_sex_fc/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
.u_sex_fc/kernel/Initializer/random_uniform/minConst*"
_class
loc:@u_sex_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
.u_sex_fc/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@u_sex_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
8u_sex_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform0u_sex_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*"
_class
loc:@u_sex_fc/kernel*
dtype0*
seed2?*
_output_shapes

:


?
.u_sex_fc/kernel/Initializer/random_uniform/subSub.u_sex_fc/kernel/Initializer/random_uniform/max.u_sex_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes
: 
?
.u_sex_fc/kernel/Initializer/random_uniform/mulMul8u_sex_fc/kernel/Initializer/random_uniform/RandomUniform.u_sex_fc/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes

:


?
*u_sex_fc/kernel/Initializer/random_uniformAdd.u_sex_fc/kernel/Initializer/random_uniform/mul.u_sex_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes

:


?
u_sex_fc/kernel
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_sex_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_sex_fc/kernel/AssignAssignu_sex_fc/kernel*u_sex_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


~
u_sex_fc/kernel/readIdentityu_sex_fc/kernel*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes

:


?
u_sex_fc/bias/Initializer/zerosConst* 
_class
loc:@u_sex_fc/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_sex_fc/bias
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_sex_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_sex_fc/bias/AssignAssignu_sex_fc/biasu_sex_fc/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

t
u_sex_fc/bias/readIdentityu_sex_fc/bias*
T0* 
_class
loc:@u_sex_fc/bias*
_output_shapes
:

i
user_fc/u_sex_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
p
user_fc/u_sex_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
 user_fc/u_sex_fc/Tensordot/ShapeShape'user_embedding/u_sex_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
j
(user_fc/u_sex_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_sex_fc/Tensordot/GatherV2GatherV2 user_fc/u_sex_fc/Tensordot/Shapeuser_fc/u_sex_fc/Tensordot/free(user_fc/u_sex_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
l
*user_fc/u_sex_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
%user_fc/u_sex_fc/Tensordot/GatherV2_1GatherV2 user_fc/u_sex_fc/Tensordot/Shapeuser_fc/u_sex_fc/Tensordot/axes*user_fc/u_sex_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
j
 user_fc/u_sex_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
user_fc/u_sex_fc/Tensordot/ProdProd#user_fc/u_sex_fc/Tensordot/GatherV2 user_fc/u_sex_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
l
"user_fc/u_sex_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
!user_fc/u_sex_fc/Tensordot/Prod_1Prod%user_fc/u_sex_fc/Tensordot/GatherV2_1"user_fc/u_sex_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
&user_fc/u_sex_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
!user_fc/u_sex_fc/Tensordot/concatConcatV2user_fc/u_sex_fc/Tensordot/freeuser_fc/u_sex_fc/Tensordot/axes&user_fc/u_sex_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
 user_fc/u_sex_fc/Tensordot/stackPackuser_fc/u_sex_fc/Tensordot/Prod!user_fc/u_sex_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
$user_fc/u_sex_fc/Tensordot/transpose	Transpose'user_embedding/u_sex_emb_layer/Identity!user_fc/u_sex_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
"user_fc/u_sex_fc/Tensordot/ReshapeReshape$user_fc/u_sex_fc/Tensordot/transpose user_fc/u_sex_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
|
+user_fc/u_sex_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
&user_fc/u_sex_fc/Tensordot/transpose_1	Transposeu_sex_fc/kernel/read+user_fc/u_sex_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


{
*user_fc/u_sex_fc/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
$user_fc/u_sex_fc/Tensordot/Reshape_1Reshape&user_fc/u_sex_fc/Tensordot/transpose_1*user_fc/u_sex_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
!user_fc/u_sex_fc/Tensordot/MatMulMatMul"user_fc/u_sex_fc/Tensordot/Reshape$user_fc/u_sex_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

l
"user_fc/u_sex_fc/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
j
(user_fc/u_sex_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_sex_fc/Tensordot/concat_1ConcatV2#user_fc/u_sex_fc/Tensordot/GatherV2"user_fc/u_sex_fc/Tensordot/Const_2(user_fc/u_sex_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_sex_fc/TensordotReshape!user_fc/u_sex_fc/Tensordot/MatMul#user_fc/u_sex_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_sex_fc/BiasAddBiasAdduser_fc/u_sex_fc/Tensordotu_sex_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

m
user_fc/u_sex_fc/ReluReluuser_fc/u_sex_fc/BiasAdd*
T0*+
_output_shapes
:?????????

?
0u_org_fc/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@u_org_fc/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
.u_org_fc/kernel/Initializer/random_uniform/minConst*"
_class
loc:@u_org_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
.u_org_fc/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@u_org_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
8u_org_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform0u_org_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*"
_class
loc:@u_org_fc/kernel*
dtype0*
seed2?*
_output_shapes

:


?
.u_org_fc/kernel/Initializer/random_uniform/subSub.u_org_fc/kernel/Initializer/random_uniform/max.u_org_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes
: 
?
.u_org_fc/kernel/Initializer/random_uniform/mulMul8u_org_fc/kernel/Initializer/random_uniform/RandomUniform.u_org_fc/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes

:


?
*u_org_fc/kernel/Initializer/random_uniformAdd.u_org_fc/kernel/Initializer/random_uniform/mul.u_org_fc/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes

:


?
u_org_fc/kernel
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_org_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_org_fc/kernel/AssignAssignu_org_fc/kernel*u_org_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


~
u_org_fc/kernel/readIdentityu_org_fc/kernel*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes

:


?
u_org_fc/bias/Initializer/zerosConst* 
_class
loc:@u_org_fc/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_org_fc/bias
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_org_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_org_fc/bias/AssignAssignu_org_fc/biasu_org_fc/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

t
u_org_fc/bias/readIdentityu_org_fc/bias*
T0* 
_class
loc:@u_org_fc/bias*
_output_shapes
:

i
user_fc/u_org_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
p
user_fc/u_org_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
 user_fc/u_org_fc/Tensordot/ShapeShape'user_embedding/u_org_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
j
(user_fc/u_org_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_org_fc/Tensordot/GatherV2GatherV2 user_fc/u_org_fc/Tensordot/Shapeuser_fc/u_org_fc/Tensordot/free(user_fc/u_org_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
l
*user_fc/u_org_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
%user_fc/u_org_fc/Tensordot/GatherV2_1GatherV2 user_fc/u_org_fc/Tensordot/Shapeuser_fc/u_org_fc/Tensordot/axes*user_fc/u_org_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
j
 user_fc/u_org_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
user_fc/u_org_fc/Tensordot/ProdProd#user_fc/u_org_fc/Tensordot/GatherV2 user_fc/u_org_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
l
"user_fc/u_org_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
!user_fc/u_org_fc/Tensordot/Prod_1Prod%user_fc/u_org_fc/Tensordot/GatherV2_1"user_fc/u_org_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
&user_fc/u_org_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
!user_fc/u_org_fc/Tensordot/concatConcatV2user_fc/u_org_fc/Tensordot/freeuser_fc/u_org_fc/Tensordot/axes&user_fc/u_org_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
 user_fc/u_org_fc/Tensordot/stackPackuser_fc/u_org_fc/Tensordot/Prod!user_fc/u_org_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
$user_fc/u_org_fc/Tensordot/transpose	Transpose'user_embedding/u_org_emb_layer/Identity!user_fc/u_org_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
"user_fc/u_org_fc/Tensordot/ReshapeReshape$user_fc/u_org_fc/Tensordot/transpose user_fc/u_org_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
|
+user_fc/u_org_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
&user_fc/u_org_fc/Tensordot/transpose_1	Transposeu_org_fc/kernel/read+user_fc/u_org_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


{
*user_fc/u_org_fc/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
$user_fc/u_org_fc/Tensordot/Reshape_1Reshape&user_fc/u_org_fc/Tensordot/transpose_1*user_fc/u_org_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
!user_fc/u_org_fc/Tensordot/MatMulMatMul"user_fc/u_org_fc/Tensordot/Reshape$user_fc/u_org_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

l
"user_fc/u_org_fc/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
j
(user_fc/u_org_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_org_fc/Tensordot/concat_1ConcatV2#user_fc/u_org_fc/Tensordot/GatherV2"user_fc/u_org_fc/Tensordot/Const_2(user_fc/u_org_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_org_fc/TensordotReshape!user_fc/u_org_fc/Tensordot/MatMul#user_fc/u_org_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_org_fc/BiasAddBiasAdduser_fc/u_org_fc/Tensordotu_org_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

m
user_fc/u_org_fc/ReluReluuser_fc/u_org_fc/BiasAdd*
T0*+
_output_shapes
:?????????

?
1u_seat_fc/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@u_seat_fc/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
/u_seat_fc/kernel/Initializer/random_uniform/minConst*#
_class
loc:@u_seat_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
/u_seat_fc/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@u_seat_fc/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
9u_seat_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform1u_seat_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*#
_class
loc:@u_seat_fc/kernel*
dtype0*
seed2?*
_output_shapes

:


?
/u_seat_fc/kernel/Initializer/random_uniform/subSub/u_seat_fc/kernel/Initializer/random_uniform/max/u_seat_fc/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes
: 
?
/u_seat_fc/kernel/Initializer/random_uniform/mulMul9u_seat_fc/kernel/Initializer/random_uniform/RandomUniform/u_seat_fc/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes

:


?
+u_seat_fc/kernel/Initializer/random_uniformAdd/u_seat_fc/kernel/Initializer/random_uniform/mul/u_seat_fc/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes

:


?
u_seat_fc/kernel
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_seat_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_seat_fc/kernel/AssignAssignu_seat_fc/kernel+u_seat_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_seat_fc/kernel/readIdentityu_seat_fc/kernel*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes

:


?
 u_seat_fc/bias/Initializer/zerosConst*!
_class
loc:@u_seat_fc/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_seat_fc/bias
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_seat_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_seat_fc/bias/AssignAssignu_seat_fc/bias u_seat_fc/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

w
u_seat_fc/bias/readIdentityu_seat_fc/bias*
T0*!
_class
loc:@u_seat_fc/bias*
_output_shapes
:

j
 user_fc/u_seat_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
q
 user_fc/u_seat_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
!user_fc/u_seat_fc/Tensordot/ShapeShape(user_embedding/u_seat_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
k
)user_fc/u_seat_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
$user_fc/u_seat_fc/Tensordot/GatherV2GatherV2!user_fc/u_seat_fc/Tensordot/Shape user_fc/u_seat_fc/Tensordot/free)user_fc/u_seat_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
m
+user_fc/u_seat_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
&user_fc/u_seat_fc/Tensordot/GatherV2_1GatherV2!user_fc/u_seat_fc/Tensordot/Shape user_fc/u_seat_fc/Tensordot/axes+user_fc/u_seat_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
k
!user_fc/u_seat_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
 user_fc/u_seat_fc/Tensordot/ProdProd$user_fc/u_seat_fc/Tensordot/GatherV2!user_fc/u_seat_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
m
#user_fc/u_seat_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
"user_fc/u_seat_fc/Tensordot/Prod_1Prod&user_fc/u_seat_fc/Tensordot/GatherV2_1#user_fc/u_seat_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
i
'user_fc/u_seat_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
"user_fc/u_seat_fc/Tensordot/concatConcatV2 user_fc/u_seat_fc/Tensordot/free user_fc/u_seat_fc/Tensordot/axes'user_fc/u_seat_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
!user_fc/u_seat_fc/Tensordot/stackPack user_fc/u_seat_fc/Tensordot/Prod"user_fc/u_seat_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
%user_fc/u_seat_fc/Tensordot/transpose	Transpose(user_embedding/u_seat_emb_layer/Identity"user_fc/u_seat_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
#user_fc/u_seat_fc/Tensordot/ReshapeReshape%user_fc/u_seat_fc/Tensordot/transpose!user_fc/u_seat_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
}
,user_fc/u_seat_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
'user_fc/u_seat_fc/Tensordot/transpose_1	Transposeu_seat_fc/kernel/read,user_fc/u_seat_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


|
+user_fc/u_seat_fc/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
%user_fc/u_seat_fc/Tensordot/Reshape_1Reshape'user_fc/u_seat_fc/Tensordot/transpose_1+user_fc/u_seat_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
"user_fc/u_seat_fc/Tensordot/MatMulMatMul#user_fc/u_seat_fc/Tensordot/Reshape%user_fc/u_seat_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

m
#user_fc/u_seat_fc/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
k
)user_fc/u_seat_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
$user_fc/u_seat_fc/Tensordot/concat_1ConcatV2$user_fc/u_seat_fc/Tensordot/GatherV2#user_fc/u_seat_fc/Tensordot/Const_2)user_fc/u_seat_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_seat_fc/TensordotReshape"user_fc/u_seat_fc/Tensordot/MatMul$user_fc/u_seat_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_seat_fc/BiasAddBiasAdduser_fc/u_seat_fc/Tensordotu_seat_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

o
user_fc/u_seat_fc/ReluReluuser_fc/u_seat_fc/BiasAdd*
T0*+
_output_shapes
:?????????

?
0u_pos_id/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@u_pos_id/kernel*
valueB"
   
   *
dtype0*
_output_shapes
:
?
.u_pos_id/kernel/Initializer/random_uniform/minConst*"
_class
loc:@u_pos_id/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
.u_pos_id/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@u_pos_id/kernel*
valueB
 *?7?*
dtype0*
_output_shapes
: 
?
8u_pos_id/kernel/Initializer/random_uniform/RandomUniformRandomUniform0u_pos_id/kernel/Initializer/random_uniform/shape*
seed?*
T0*"
_class
loc:@u_pos_id/kernel*
dtype0*
seed2?*
_output_shapes

:


?
.u_pos_id/kernel/Initializer/random_uniform/subSub.u_pos_id/kernel/Initializer/random_uniform/max.u_pos_id/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes
: 
?
.u_pos_id/kernel/Initializer/random_uniform/mulMul8u_pos_id/kernel/Initializer/random_uniform/RandomUniform.u_pos_id/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes

:


?
*u_pos_id/kernel/Initializer/random_uniformAdd.u_pos_id/kernel/Initializer/random_uniform/mul.u_pos_id/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes

:


?
u_pos_id/kernel
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_pos_id/kernel*
dtype0*
	container *
_output_shapes

:


?
u_pos_id/kernel/AssignAssignu_pos_id/kernel*u_pos_id/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


~
u_pos_id/kernel/readIdentityu_pos_id/kernel*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes

:


?
u_pos_id/bias/Initializer/zerosConst* 
_class
loc:@u_pos_id/bias*
valueB
*    *
dtype0*
_output_shapes
:

?
u_pos_id/bias
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_pos_id/bias*
dtype0*
	container *
_output_shapes
:

?
u_pos_id/bias/AssignAssignu_pos_id/biasu_pos_id/bias/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

t
u_pos_id/bias/readIdentityu_pos_id/bias*
T0* 
_class
loc:@u_pos_id/bias*
_output_shapes
:

i
user_fc/u_pos_id/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
p
user_fc/u_pos_id/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
 user_fc/u_pos_id/Tensordot/ShapeShape'user_embedding/u_pos_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
j
(user_fc/u_pos_id/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_pos_id/Tensordot/GatherV2GatherV2 user_fc/u_pos_id/Tensordot/Shapeuser_fc/u_pos_id/Tensordot/free(user_fc/u_pos_id/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
l
*user_fc/u_pos_id/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
%user_fc/u_pos_id/Tensordot/GatherV2_1GatherV2 user_fc/u_pos_id/Tensordot/Shapeuser_fc/u_pos_id/Tensordot/axes*user_fc/u_pos_id/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
j
 user_fc/u_pos_id/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
user_fc/u_pos_id/Tensordot/ProdProd#user_fc/u_pos_id/Tensordot/GatherV2 user_fc/u_pos_id/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
l
"user_fc/u_pos_id/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
!user_fc/u_pos_id/Tensordot/Prod_1Prod%user_fc/u_pos_id/Tensordot/GatherV2_1"user_fc/u_pos_id/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
&user_fc/u_pos_id/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
!user_fc/u_pos_id/Tensordot/concatConcatV2user_fc/u_pos_id/Tensordot/freeuser_fc/u_pos_id/Tensordot/axes&user_fc/u_pos_id/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
 user_fc/u_pos_id/Tensordot/stackPackuser_fc/u_pos_id/Tensordot/Prod!user_fc/u_pos_id/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
$user_fc/u_pos_id/Tensordot/transpose	Transpose'user_embedding/u_pos_emb_layer/Identity!user_fc/u_pos_id/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????

?
"user_fc/u_pos_id/Tensordot/ReshapeReshape$user_fc/u_pos_id/Tensordot/transpose user_fc/u_pos_id/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
|
+user_fc/u_pos_id/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
&user_fc/u_pos_id/Tensordot/transpose_1	Transposeu_pos_id/kernel/read+user_fc/u_pos_id/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:


{
*user_fc/u_pos_id/Tensordot/Reshape_1/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
$user_fc/u_pos_id/Tensordot/Reshape_1Reshape&user_fc/u_pos_id/Tensordot/transpose_1*user_fc/u_pos_id/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:


?
!user_fc/u_pos_id/Tensordot/MatMulMatMul"user_fc/u_pos_id/Tensordot/Reshape$user_fc/u_pos_id/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????

l
"user_fc/u_pos_id/Tensordot/Const_2Const*
valueB:
*
dtype0*
_output_shapes
:
j
(user_fc/u_pos_id/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
#user_fc/u_pos_id/Tensordot/concat_1ConcatV2#user_fc/u_pos_id/Tensordot/GatherV2"user_fc/u_pos_id/Tensordot/Const_2(user_fc/u_pos_id/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_pos_id/TensordotReshape!user_fc/u_pos_id/Tensordot/MatMul#user_fc/u_pos_id/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????

?
user_fc/u_pos_id/BiasAddBiasAdduser_fc/u_pos_id/Tensordotu_pos_id/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????

m
user_fc/u_pos_id/ReluReluuser_fc/u_pos_id/BiasAdd*
T0*+
_output_shapes
:?????????

Y
user_fc/u_concated/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
user_fc/u_concatedConcatV2user_fc/u_type_fc/Reluuser_fc/u_age_fc/Reluuser_fc/u_sex_fc/Reluuser_fc/u_org_fc/Reluuser_fc/u_seat_fc/Reluuser_fc/u_pos_id/Reluuser_fc/u_concated/axis*

Tidx0*
T0*
N*+
_output_shapes
:?????????<
?
5u_concated_fc/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@u_concated_fc/kernel*
valueB"<   @   *
dtype0*
_output_shapes
:
?
3u_concated_fc/kernel/Initializer/random_uniform/minConst*'
_class
loc:@u_concated_fc/kernel*
valueB
 *??a?*
dtype0*
_output_shapes
: 
?
3u_concated_fc/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@u_concated_fc/kernel*
valueB
 *??a>*
dtype0*
_output_shapes
: 
?
=u_concated_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform5u_concated_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*'
_class
loc:@u_concated_fc/kernel*
dtype0*
seed2?*
_output_shapes

:<@
?
3u_concated_fc/kernel/Initializer/random_uniform/subSub3u_concated_fc/kernel/Initializer/random_uniform/max3u_concated_fc/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes
: 
?
3u_concated_fc/kernel/Initializer/random_uniform/mulMul=u_concated_fc/kernel/Initializer/random_uniform/RandomUniform3u_concated_fc/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
/u_concated_fc/kernel/Initializer/random_uniformAdd3u_concated_fc/kernel/Initializer/random_uniform/mul3u_concated_fc/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
u_concated_fc/kernel
VariableV2*
shape
:<@*
shared_name *'
_class
loc:@u_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:<@
?
u_concated_fc/kernel/AssignAssignu_concated_fc/kernel/u_concated_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
u_concated_fc/kernel/readIdentityu_concated_fc/kernel*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
$u_concated_fc/bias/Initializer/zerosConst*%
_class
loc:@u_concated_fc/bias*
valueB@*    *
dtype0*
_output_shapes
:@
?
u_concated_fc/bias
VariableV2*
shape:@*
shared_name *%
_class
loc:@u_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
u_concated_fc/bias/AssignAssignu_concated_fc/bias$u_concated_fc/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
u_concated_fc/bias/readIdentityu_concated_fc/bias*
T0*%
_class
loc:@u_concated_fc/bias*
_output_shapes
:@
n
$user_fc/u_concated_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
u
$user_fc/u_concated_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
w
%user_fc/u_concated_fc/Tensordot/ShapeShapeuser_fc/u_concated*
T0*
out_type0*
_output_shapes
:
o
-user_fc/u_concated_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
(user_fc/u_concated_fc/Tensordot/GatherV2GatherV2%user_fc/u_concated_fc/Tensordot/Shape$user_fc/u_concated_fc/Tensordot/free-user_fc/u_concated_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
q
/user_fc/u_concated_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
*user_fc/u_concated_fc/Tensordot/GatherV2_1GatherV2%user_fc/u_concated_fc/Tensordot/Shape$user_fc/u_concated_fc/Tensordot/axes/user_fc/u_concated_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
o
%user_fc/u_concated_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
$user_fc/u_concated_fc/Tensordot/ProdProd(user_fc/u_concated_fc/Tensordot/GatherV2%user_fc/u_concated_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
q
'user_fc/u_concated_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
&user_fc/u_concated_fc/Tensordot/Prod_1Prod*user_fc/u_concated_fc/Tensordot/GatherV2_1'user_fc/u_concated_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
m
+user_fc/u_concated_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
&user_fc/u_concated_fc/Tensordot/concatConcatV2$user_fc/u_concated_fc/Tensordot/free$user_fc/u_concated_fc/Tensordot/axes+user_fc/u_concated_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
%user_fc/u_concated_fc/Tensordot/stackPack$user_fc/u_concated_fc/Tensordot/Prod&user_fc/u_concated_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
)user_fc/u_concated_fc/Tensordot/transpose	Transposeuser_fc/u_concated&user_fc/u_concated_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????<
?
'user_fc/u_concated_fc/Tensordot/ReshapeReshape)user_fc/u_concated_fc/Tensordot/transpose%user_fc/u_concated_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
0user_fc/u_concated_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
+user_fc/u_concated_fc/Tensordot/transpose_1	Transposeu_concated_fc/kernel/read0user_fc/u_concated_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:<@
?
/user_fc/u_concated_fc/Tensordot/Reshape_1/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
?
)user_fc/u_concated_fc/Tensordot/Reshape_1Reshape+user_fc/u_concated_fc/Tensordot/transpose_1/user_fc/u_concated_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:<@
?
&user_fc/u_concated_fc/Tensordot/MatMulMatMul'user_fc/u_concated_fc/Tensordot/Reshape)user_fc/u_concated_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????@
q
'user_fc/u_concated_fc/Tensordot/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
o
-user_fc/u_concated_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
(user_fc/u_concated_fc/Tensordot/concat_1ConcatV2(user_fc/u_concated_fc/Tensordot/GatherV2'user_fc/u_concated_fc/Tensordot/Const_2-user_fc/u_concated_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
user_fc/u_concated_fc/TensordotReshape&user_fc/u_concated_fc/Tensordot/MatMul(user_fc/u_concated_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????@
?
user_fc/u_concated_fc/BiasAddBiasAdduser_fc/u_concated_fc/Tensordotu_concated_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????@
w
user_fc/u_concated_fc/ReluReluuser_fc/u_concated_fc/BiasAdd*
T0*+
_output_shapes
:?????????@
z
)item_class_embedding/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
l
'item_class_embedding/random_uniform/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
l
'item_class_embedding/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1item_class_embedding/random_uniform/RandomUniformRandomUniform)item_class_embedding/random_uniform/shape*
seed?*
T0*
dtype0*
seed2?*
_output_shapes

:
?
'item_class_embedding/random_uniform/subSub'item_class_embedding/random_uniform/max'item_class_embedding/random_uniform/min*
T0*
_output_shapes
: 
?
'item_class_embedding/random_uniform/mulMul1item_class_embedding/random_uniform/RandomUniform'item_class_embedding/random_uniform/sub*
T0*
_output_shapes

:
?
#item_class_embedding/random_uniformAdd'item_class_embedding/random_uniform/mul'item_class_embedding/random_uniform/min*
T0*
_output_shapes

:
?
'item_class_embedding/i_class_emb_matrix
VariableV2*
shape
:*
shared_name *
dtype0*
	container *
_output_shapes

:
?
.item_class_embedding/i_class_emb_matrix/AssignAssign'item_class_embedding/i_class_emb_matrix#item_class_embedding/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
,item_class_embedding/i_class_emb_matrix/readIdentity'item_class_embedding/i_class_emb_matrix*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
+item_class_embedding/i_class_emb_layer/axisConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
value	B : *
dtype0*
_output_shapes
: 
?
&item_class_embedding/i_class_emb_layerGatherV2,item_class_embedding/i_class_emb_matrix/readi_class_label+item_class_embedding/i_class_emb_layer/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*+
_output_shapes
:?????????
?
/item_class_embedding/i_class_emb_layer/IdentityIdentity&item_class_embedding/i_class_emb_layer*
T0*+
_output_shapes
:?????????
?
2i_class_fc/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@i_class_fc/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
0i_class_fc/kernel/Initializer/random_uniform/minConst*$
_class
loc:@i_class_fc/kernel*
valueB
 *׳ݾ*
dtype0*
_output_shapes
: 
?
0i_class_fc/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@i_class_fc/kernel*
valueB
 *׳?>*
dtype0*
_output_shapes
: 
?
:i_class_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform2i_class_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*$
_class
loc:@i_class_fc/kernel*
dtype0*
seed2?*
_output_shapes

:
?
0i_class_fc/kernel/Initializer/random_uniform/subSub0i_class_fc/kernel/Initializer/random_uniform/max0i_class_fc/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes
: 
?
0i_class_fc/kernel/Initializer/random_uniform/mulMul:i_class_fc/kernel/Initializer/random_uniform/RandomUniform0i_class_fc/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes

:
?
,i_class_fc/kernel/Initializer/random_uniformAdd0i_class_fc/kernel/Initializer/random_uniform/mul0i_class_fc/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes

:
?
i_class_fc/kernel
VariableV2*
shape
:*
shared_name *$
_class
loc:@i_class_fc/kernel*
dtype0*
	container *
_output_shapes

:
?
i_class_fc/kernel/AssignAssigni_class_fc/kernel,i_class_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
i_class_fc/kernel/readIdentityi_class_fc/kernel*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes

:
?
!i_class_fc/bias/Initializer/zerosConst*"
_class
loc:@i_class_fc/bias*
valueB*    *
dtype0*
_output_shapes
:
?
i_class_fc/bias
VariableV2*
shape:*
shared_name *"
_class
loc:@i_class_fc/bias*
dtype0*
	container *
_output_shapes
:
?
i_class_fc/bias/AssignAssigni_class_fc/bias!i_class_fc/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
z
i_class_fc/bias/readIdentityi_class_fc/bias*
T0*"
_class
loc:@i_class_fc/bias*
_output_shapes
:
q
'item_class_fc/i_class_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
x
'item_class_fc/i_class_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
(item_class_fc/i_class_fc/Tensordot/ShapeShape/item_class_embedding/i_class_emb_layer/Identity*
T0*
out_type0*
_output_shapes
:
r
0item_class_fc/i_class_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
+item_class_fc/i_class_fc/Tensordot/GatherV2GatherV2(item_class_fc/i_class_fc/Tensordot/Shape'item_class_fc/i_class_fc/Tensordot/free0item_class_fc/i_class_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
t
2item_class_fc/i_class_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
-item_class_fc/i_class_fc/Tensordot/GatherV2_1GatherV2(item_class_fc/i_class_fc/Tensordot/Shape'item_class_fc/i_class_fc/Tensordot/axes2item_class_fc/i_class_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
r
(item_class_fc/i_class_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
'item_class_fc/i_class_fc/Tensordot/ProdProd+item_class_fc/i_class_fc/Tensordot/GatherV2(item_class_fc/i_class_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
*item_class_fc/i_class_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
)item_class_fc/i_class_fc/Tensordot/Prod_1Prod-item_class_fc/i_class_fc/Tensordot/GatherV2_1*item_class_fc/i_class_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
.item_class_fc/i_class_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
)item_class_fc/i_class_fc/Tensordot/concatConcatV2'item_class_fc/i_class_fc/Tensordot/free'item_class_fc/i_class_fc/Tensordot/axes.item_class_fc/i_class_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
(item_class_fc/i_class_fc/Tensordot/stackPack'item_class_fc/i_class_fc/Tensordot/Prod)item_class_fc/i_class_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
,item_class_fc/i_class_fc/Tensordot/transpose	Transpose/item_class_embedding/i_class_emb_layer/Identity)item_class_fc/i_class_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????
?
*item_class_fc/i_class_fc/Tensordot/ReshapeReshape,item_class_fc/i_class_fc/Tensordot/transpose(item_class_fc/i_class_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
3item_class_fc/i_class_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
.item_class_fc/i_class_fc/Tensordot/transpose_1	Transposei_class_fc/kernel/read3item_class_fc/i_class_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:
?
2item_class_fc/i_class_fc/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
,item_class_fc/i_class_fc/Tensordot/Reshape_1Reshape.item_class_fc/i_class_fc/Tensordot/transpose_12item_class_fc/i_class_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:
?
)item_class_fc/i_class_fc/Tensordot/MatMulMatMul*item_class_fc/i_class_fc/Tensordot/Reshape,item_class_fc/i_class_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
t
*item_class_fc/i_class_fc/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
r
0item_class_fc/i_class_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
+item_class_fc/i_class_fc/Tensordot/concat_1ConcatV2+item_class_fc/i_class_fc/Tensordot/GatherV2*item_class_fc/i_class_fc/Tensordot/Const_20item_class_fc/i_class_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
"item_class_fc/i_class_fc/TensordotReshape)item_class_fc/i_class_fc/Tensordot/MatMul+item_class_fc/i_class_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
 item_class_fc/i_class_fc/BiasAddBiasAdd"item_class_fc/i_class_fc/Tensordoti_class_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????
}
item_class_fc/i_class_fc/ReluRelu item_class_fc/i_class_fc/BiasAdd*
T0*+
_output_shapes
:?????????
?
9i_entities_emb_fc/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@i_entities_emb_fc/kernel*
valueB"@       *
dtype0*
_output_shapes
:
?
7i_entities_emb_fc/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@i_entities_emb_fc/kernel*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7i_entities_emb_fc/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@i_entities_emb_fc/kernel*
valueB
 *  ?>*
dtype0*
_output_shapes
: 
?
Ai_entities_emb_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform9i_entities_emb_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
dtype0*
seed2?*
_output_shapes

:@ 
?
7i_entities_emb_fc/kernel/Initializer/random_uniform/subSub7i_entities_emb_fc/kernel/Initializer/random_uniform/max7i_entities_emb_fc/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
_output_shapes
: 
?
7i_entities_emb_fc/kernel/Initializer/random_uniform/mulMulAi_entities_emb_fc/kernel/Initializer/random_uniform/RandomUniform7i_entities_emb_fc/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
_output_shapes

:@ 
?
3i_entities_emb_fc/kernel/Initializer/random_uniformAdd7i_entities_emb_fc/kernel/Initializer/random_uniform/mul7i_entities_emb_fc/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
_output_shapes

:@ 
?
i_entities_emb_fc/kernel
VariableV2*
shape
:@ *
shared_name *+
_class!
loc:@i_entities_emb_fc/kernel*
dtype0*
	container *
_output_shapes

:@ 
?
i_entities_emb_fc/kernel/AssignAssigni_entities_emb_fc/kernel3i_entities_emb_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
validate_shape(*
_output_shapes

:@ 
?
i_entities_emb_fc/kernel/readIdentityi_entities_emb_fc/kernel*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
_output_shapes

:@ 
?
(i_entities_emb_fc/bias/Initializer/zerosConst*)
_class
loc:@i_entities_emb_fc/bias*
valueB *    *
dtype0*
_output_shapes
: 
?
i_entities_emb_fc/bias
VariableV2*
shape: *
shared_name *)
_class
loc:@i_entities_emb_fc/bias*
dtype0*
	container *
_output_shapes
: 
?
i_entities_emb_fc/bias/AssignAssigni_entities_emb_fc/bias(i_entities_emb_fc/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@i_entities_emb_fc/bias*
validate_shape(*
_output_shapes
: 
?
i_entities_emb_fc/bias/readIdentityi_entities_emb_fc/bias*
T0*)
_class
loc:@i_entities_emb_fc/bias*
_output_shapes
: 
?
+item_wv_dim_reduce/i_entities_emb_fc/MatMulMatMul
i_entitiesi_entities_emb_fc/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
?
,item_wv_dim_reduce/i_entities_emb_fc/BiasAddBiasAdd+item_wv_dim_reduce/i_entities_emb_fc/MatMuli_entities_emb_fc/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:????????? 
?
)item_wv_dim_reduce/i_entities_emb_fc/ReluRelu,item_wv_dim_reduce/i_entities_emb_fc/BiasAdd*
T0*'
_output_shapes
:????????? 
?
5i_concated_fc/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@i_concated_fc/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
?
3i_concated_fc/kernel/Initializer/random_uniform/minConst*'
_class
loc:@i_concated_fc/kernel*
valueB
 *?7??*
dtype0*
_output_shapes
: 
?
3i_concated_fc/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@i_concated_fc/kernel*
valueB
 *?7?>*
dtype0*
_output_shapes
: 
?
=i_concated_fc/kernel/Initializer/random_uniform/RandomUniformRandomUniform5i_concated_fc/kernel/Initializer/random_uniform/shape*
seed?*
T0*'
_class
loc:@i_concated_fc/kernel*
dtype0*
seed2?*
_output_shapes

:@
?
3i_concated_fc/kernel/Initializer/random_uniform/subSub3i_concated_fc/kernel/Initializer/random_uniform/max3i_concated_fc/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes
: 
?
3i_concated_fc/kernel/Initializer/random_uniform/mulMul=i_concated_fc/kernel/Initializer/random_uniform/RandomUniform3i_concated_fc/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
/i_concated_fc/kernel/Initializer/random_uniformAdd3i_concated_fc/kernel/Initializer/random_uniform/mul3i_concated_fc/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
i_concated_fc/kernel
VariableV2*
shape
:@*
shared_name *'
_class
loc:@i_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:@
?
i_concated_fc/kernel/AssignAssigni_concated_fc/kernel/i_concated_fc/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
i_concated_fc/kernel/readIdentityi_concated_fc/kernel*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
$i_concated_fc/bias/Initializer/zerosConst*%
_class
loc:@i_concated_fc/bias*
valueB@*    *
dtype0*
_output_shapes
:@
?
i_concated_fc/bias
VariableV2*
shape:@*
shared_name *%
_class
loc:@i_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
i_concated_fc/bias/AssignAssigni_concated_fc/bias$i_concated_fc/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
i_concated_fc/bias/readIdentityi_concated_fc/bias*
T0*%
_class
loc:@i_concated_fc/bias*
_output_shapes
:@
w
-item_concated_fc/i_concated_fc/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
~
-item_concated_fc/i_concated_fc/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
?
.item_concated_fc/i_concated_fc/Tensordot/ShapeShapeitem_class_fc/i_class_fc/Relu*
T0*
out_type0*
_output_shapes
:
x
6item_concated_fc/i_concated_fc/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
1item_concated_fc/i_concated_fc/Tensordot/GatherV2GatherV2.item_concated_fc/i_concated_fc/Tensordot/Shape-item_concated_fc/i_concated_fc/Tensordot/free6item_concated_fc/i_concated_fc/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
z
8item_concated_fc/i_concated_fc/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
3item_concated_fc/i_concated_fc/Tensordot/GatherV2_1GatherV2.item_concated_fc/i_concated_fc/Tensordot/Shape-item_concated_fc/i_concated_fc/Tensordot/axes8item_concated_fc/i_concated_fc/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
x
.item_concated_fc/i_concated_fc/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
-item_concated_fc/i_concated_fc/Tensordot/ProdProd1item_concated_fc/i_concated_fc/Tensordot/GatherV2.item_concated_fc/i_concated_fc/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
z
0item_concated_fc/i_concated_fc/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
/item_concated_fc/i_concated_fc/Tensordot/Prod_1Prod3item_concated_fc/i_concated_fc/Tensordot/GatherV2_10item_concated_fc/i_concated_fc/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
v
4item_concated_fc/i_concated_fc/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
/item_concated_fc/i_concated_fc/Tensordot/concatConcatV2-item_concated_fc/i_concated_fc/Tensordot/free-item_concated_fc/i_concated_fc/Tensordot/axes4item_concated_fc/i_concated_fc/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
.item_concated_fc/i_concated_fc/Tensordot/stackPack-item_concated_fc/i_concated_fc/Tensordot/Prod/item_concated_fc/i_concated_fc/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
2item_concated_fc/i_concated_fc/Tensordot/transpose	Transposeitem_class_fc/i_class_fc/Relu/item_concated_fc/i_concated_fc/Tensordot/concat*
Tperm0*
T0*+
_output_shapes
:?????????
?
0item_concated_fc/i_concated_fc/Tensordot/ReshapeReshape2item_concated_fc/i_concated_fc/Tensordot/transpose.item_concated_fc/i_concated_fc/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
9item_concated_fc/i_concated_fc/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
4item_concated_fc/i_concated_fc/Tensordot/transpose_1	Transposei_concated_fc/kernel/read9item_concated_fc/i_concated_fc/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:@
?
8item_concated_fc/i_concated_fc/Tensordot/Reshape_1/shapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
?
2item_concated_fc/i_concated_fc/Tensordot/Reshape_1Reshape4item_concated_fc/i_concated_fc/Tensordot/transpose_18item_concated_fc/i_concated_fc/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:@
?
/item_concated_fc/i_concated_fc/Tensordot/MatMulMatMul0item_concated_fc/i_concated_fc/Tensordot/Reshape2item_concated_fc/i_concated_fc/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????@
z
0item_concated_fc/i_concated_fc/Tensordot/Const_2Const*
valueB:@*
dtype0*
_output_shapes
:
x
6item_concated_fc/i_concated_fc/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
1item_concated_fc/i_concated_fc/Tensordot/concat_1ConcatV21item_concated_fc/i_concated_fc/Tensordot/GatherV20item_concated_fc/i_concated_fc/Tensordot/Const_26item_concated_fc/i_concated_fc/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
(item_concated_fc/i_concated_fc/TensordotReshape/item_concated_fc/i_concated_fc/Tensordot/MatMul1item_concated_fc/i_concated_fc/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????@
?
&item_concated_fc/i_concated_fc/BiasAddBiasAdd(item_concated_fc/i_concated_fc/Tensordoti_concated_fc/bias/read*
T0*
data_formatNHWC*+
_output_shapes
:?????????@
?
#item_concated_fc/i_concated_fc/ReluRelu&item_concated_fc/i_concated_fc/BiasAdd*
T0*+
_output_shapes
:?????????@
T
hidden/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
hidden/concatConcatV2user_fc/u_concated_fc/Relu#item_concated_fc/i_concated_fc/Reluhidden/concat/axis*

Tidx0*
T0*
N*,
_output_shapes
:??????????
e
hidden/Reshape/shapeConst*
valueB"?????   *
dtype0*
_output_shapes
:

hidden/ReshapeReshapehidden/concathidden/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
0expert_weight/Initializer/truncated_normal/shapeConst* 
_class
loc:@expert_weight*!
valueB"?         *
dtype0*
_output_shapes
:
?
/expert_weight/Initializer/truncated_normal/meanConst* 
_class
loc:@expert_weight*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1expert_weight/Initializer/truncated_normal/stddevConst* 
_class
loc:@expert_weight*
valueB
 *???<*
dtype0*
_output_shapes
: 
?
:expert_weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0expert_weight/Initializer/truncated_normal/shape*
seed?*
T0* 
_class
loc:@expert_weight*
dtype0*
seed2?*#
_output_shapes
:?
?
.expert_weight/Initializer/truncated_normal/mulMul:expert_weight/Initializer/truncated_normal/TruncatedNormal1expert_weight/Initializer/truncated_normal/stddev*
T0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
*expert_weight/Initializer/truncated_normalAdd.expert_weight/Initializer/truncated_normal/mul/expert_weight/Initializer/truncated_normal/mean*
T0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
expert_weight
VariableV2*
shape:?*
shared_name * 
_class
loc:@expert_weight*
dtype0*
	container *#
_output_shapes
:?
?
expert_weight/AssignAssignexpert_weight*expert_weight/Initializer/truncated_normal*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
}
expert_weight/readIdentityexpert_weight*
T0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
expert_bias/Initializer/zerosConst*
_class
loc:@expert_bias*
valueB*    *
dtype0*
_output_shapes
:
?
expert_bias
VariableV2*
shape:*
shared_name *
_class
loc:@expert_bias*
dtype0*
	container *
_output_shapes
:
?
expert_bias/AssignAssignexpert_biasexpert_bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
n
expert_bias/readIdentityexpert_bias*
T0*
_class
loc:@expert_bias*
_output_shapes
:
_
expert/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
_
expert/Tensordot/freeConst*
valueB: *
dtype0*
_output_shapes
:
d
expert/Tensordot/ShapeShapehidden/Reshape*
T0*
out_type0*
_output_shapes
:
`
expert/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
expert/Tensordot/GatherV2GatherV2expert/Tensordot/Shapeexpert/Tensordot/freeexpert/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
b
 expert/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
expert/Tensordot/GatherV2_1GatherV2expert/Tensordot/Shapeexpert/Tensordot/axes expert/Tensordot/GatherV2_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_output_shapes
:
`
expert/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
expert/Tensordot/ProdProdexpert/Tensordot/GatherV2expert/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
b
expert/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
expert/Tensordot/Prod_1Prodexpert/Tensordot/GatherV2_1expert/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
expert/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
expert/Tensordot/concatConcatV2expert/Tensordot/freeexpert/Tensordot/axesexpert/Tensordot/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
expert/Tensordot/stackPackexpert/Tensordot/Prodexpert/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
?
expert/Tensordot/transpose	Transposehidden/Reshapeexpert/Tensordot/concat*
Tperm0*
T0*(
_output_shapes
:??????????
?
expert/Tensordot/ReshapeReshapeexpert/Tensordot/transposeexpert/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:??????????????????
v
!expert/Tensordot/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
?
expert/Tensordot/transpose_1	Transposeexpert_weight/read!expert/Tensordot/transpose_1/perm*
Tperm0*
T0*#
_output_shapes
:?
q
 expert/Tensordot/Reshape_1/shapeConst*
valueB"?   @   *
dtype0*
_output_shapes
:
?
expert/Tensordot/Reshape_1Reshapeexpert/Tensordot/transpose_1 expert/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	?@
?
expert/Tensordot/MatMulMatMulexpert/Tensordot/Reshapeexpert/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????@
i
expert/Tensordot/Const_2Const*
valueB"      *
dtype0*
_output_shapes
:
`
expert/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
expert/Tensordot/concat_1ConcatV2expert/Tensordot/GatherV2expert/Tensordot/Const_2expert/Tensordot/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
expert/TensordotReshapeexpert/Tensordot/MatMulexpert/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:?????????
k

expert/AddAddexpert/Tensordotexpert_bias/read*
T0*+
_output_shapes
:?????????
[
expert/expert_outRelu
expert/Add*
T0*+
_output_shapes
:?????????
?
/gate1_weight/Initializer/truncated_normal/shapeConst*
_class
loc:@gate1_weight*
valueB"?      *
dtype0*
_output_shapes
:
?
.gate1_weight/Initializer/truncated_normal/meanConst*
_class
loc:@gate1_weight*
valueB
 *    *
dtype0*
_output_shapes
: 
?
0gate1_weight/Initializer/truncated_normal/stddevConst*
_class
loc:@gate1_weight*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
9gate1_weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal/gate1_weight/Initializer/truncated_normal/shape*
seed?*
T0*
_class
loc:@gate1_weight*
dtype0*
seed2?*
_output_shapes
:	?
?
-gate1_weight/Initializer/truncated_normal/mulMul9gate1_weight/Initializer/truncated_normal/TruncatedNormal0gate1_weight/Initializer/truncated_normal/stddev*
T0*
_class
loc:@gate1_weight*
_output_shapes
:	?
?
)gate1_weight/Initializer/truncated_normalAdd-gate1_weight/Initializer/truncated_normal/mul.gate1_weight/Initializer/truncated_normal/mean*
T0*
_class
loc:@gate1_weight*
_output_shapes
:	?
?
gate1_weight
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate1_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate1_weight/AssignAssigngate1_weight)gate1_weight/Initializer/truncated_normal*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
v
gate1_weight/readIdentitygate1_weight*
T0*
_class
loc:@gate1_weight*
_output_shapes
:	?
?
gate1_bias/Initializer/zerosConst*
_class
loc:@gate1_bias*
valueB*    *
dtype0*
_output_shapes
:
?

gate1_bias
VariableV2*
shape:*
shared_name *
_class
loc:@gate1_bias*
dtype0*
	container *
_output_shapes
:
?
gate1_bias/AssignAssign
gate1_biasgate1_bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
k
gate1_bias/readIdentity
gate1_bias*
T0*
_class
loc:@gate1_bias*
_output_shapes
:
?
gate1/MatMulMatMulhidden/Reshapegate1_weight/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
a
	gate1/AddAddgate1/MatMulgate1_bias/read*
T0*'
_output_shapes
:?????????
W
gate1/gate1_outSoftmax	gate1/Add*
T0*'
_output_shapes
:?????????
?
/gate2_weight/Initializer/truncated_normal/shapeConst*
_class
loc:@gate2_weight*
valueB"?      *
dtype0*
_output_shapes
:
?
.gate2_weight/Initializer/truncated_normal/meanConst*
_class
loc:@gate2_weight*
valueB
 *    *
dtype0*
_output_shapes
: 
?
0gate2_weight/Initializer/truncated_normal/stddevConst*
_class
loc:@gate2_weight*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
9gate2_weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal/gate2_weight/Initializer/truncated_normal/shape*
seed?*
T0*
_class
loc:@gate2_weight*
dtype0*
seed2?*
_output_shapes
:	?
?
-gate2_weight/Initializer/truncated_normal/mulMul9gate2_weight/Initializer/truncated_normal/TruncatedNormal0gate2_weight/Initializer/truncated_normal/stddev*
T0*
_class
loc:@gate2_weight*
_output_shapes
:	?
?
)gate2_weight/Initializer/truncated_normalAdd-gate2_weight/Initializer/truncated_normal/mul.gate2_weight/Initializer/truncated_normal/mean*
T0*
_class
loc:@gate2_weight*
_output_shapes
:	?
?
gate2_weight
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate2_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate2_weight/AssignAssigngate2_weight)gate2_weight/Initializer/truncated_normal*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
v
gate2_weight/readIdentitygate2_weight*
T0*
_class
loc:@gate2_weight*
_output_shapes
:	?
?
gate2_bias/Initializer/zerosConst*
_class
loc:@gate2_bias*
valueB*    *
dtype0*
_output_shapes
:
?

gate2_bias
VariableV2*
shape:*
shared_name *
_class
loc:@gate2_bias*
dtype0*
	container *
_output_shapes
:
?
gate2_bias/AssignAssign
gate2_biasgate2_bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
k
gate2_bias/readIdentity
gate2_bias*
T0*
_class
loc:@gate2_bias*
_output_shapes
:
?
gate2/MatMulMatMulhidden/Reshapegate2_weight/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
a
	gate2/AddAddgate2/MatMulgate2_bias/read*
T0*'
_output_shapes
:?????????
W
gate2/gate2_outSoftmax	gate2/Add*
T0*'
_output_shapes
:?????????
]
label1_input/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
label1_input/ExpandDims
ExpandDimsgate1/gate1_outlabel1_input/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:?????????
y
label1_input/MulMulexpert/expert_outlabel1_input/ExpandDims*
T0*+
_output_shapes
:?????????
d
"label1_input/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
label1_input/SumSumlabel1_input/Mul"label1_input/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*'
_output_shapes
:?????????
k
label1_input/Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
label1_input/ReshapeReshapelabel1_input/Sumlabel1_input/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *׳ݾ*
dtype0*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *׳?>*
dtype0*
_output_shapes
: 
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed?*
T0*
_class
loc:@dense/kernel*
dtype0*
seed2?*
_output_shapes

:
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/kernel
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
?
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
?

dense/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
:
?
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
label1_output/dense/MatMulMatMullabel1_input/Reshapedense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
?
label1_output/dense/BiasAddBiasAddlabel1_output/dense/MatMuldense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
o
label1_output/dense/ReluRelulabel1_output/dense/BiasAdd*
T0*'
_output_shapes
:?????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"       *
dtype0*
_output_shapes
:
?
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *???*
dtype0*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *??>*
dtype0*
_output_shapes
: 
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed?*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
seed2?*
_output_shapes

: 
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
?
dense_1/kernel
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
?
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB *    *
dtype0*
_output_shapes
: 
?
dense_1/bias
VariableV2*
shape: *
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container *
_output_shapes
: 
?
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
: 
?
label1_output/dense_1/MatMulMatMullabel1_output/dense/Reludense_1/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
?
label1_output/dense_1/BiasAddBiasAddlabel1_output/dense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:????????? 
s
label1_output/dense_1/ReluRelulabel1_output/dense_1/BiasAdd*
T0*'
_output_shapes
:????????? 
?
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
valueB"       *
dtype0*
_output_shapes
:
?
-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
valueB
 *JQھ*
dtype0*
_output_shapes
: 
?
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
valueB
 *JQ?>*
dtype0*
_output_shapes
: 
?
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed?*
T0*!
_class
loc:@dense_2/kernel*
dtype0*
seed2?*
_output_shapes

: 
?
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
?
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
?
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
?
dense_2/kernel
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
?
dense_2/bias/Initializer/zerosConst*
_class
loc:@dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
?
dense_2/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container *
_output_shapes
:
?
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
?
label1_output/dense_2/MatMulMatMullabel1_output/dense_1/Reludense_2/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
?
label1_output/dense_2/BiasAddBiasAddlabel1_output/dense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
c
SigmoidSigmoidlabel1_output/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
J
ctrIdentitySigmoid*
T0*'
_output_shapes
:?????????
]
label2_input/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
label2_input/ExpandDims
ExpandDimsgate2/gate2_outlabel2_input/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:?????????
y
label2_input/MulMulexpert/expert_outlabel2_input/ExpandDims*
T0*+
_output_shapes
:?????????
d
"label2_input/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
label2_input/SumSumlabel2_input/Mul"label2_input/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*'
_output_shapes
:?????????
k
label2_input/Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
label2_input/ReshapeReshapelabel2_input/Sumlabel2_input/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
/dense_3/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
valueB
 *׳ݾ*
dtype0*
_output_shapes
: 
?
-dense_3/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_3/kernel*
valueB
 *׳?>*
dtype0*
_output_shapes
: 
?
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
seed?*
T0*!
_class
loc:@dense_3/kernel*
dtype0*
seed2?*
_output_shapes

:
?
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
?
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
?
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
?
dense_3/kernel
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_3/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
?
dense_3/bias/Initializer/zerosConst*
_class
loc:@dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
?
dense_3/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense_3/bias*
dtype0*
	container *
_output_shapes
:
?
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
?
label2_output/dense/MatMulMatMullabel2_input/Reshapedense_3/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
?
label2_output/dense/BiasAddBiasAddlabel2_output/dense/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
o
label2_output/dense/ReluRelulabel2_output/dense/BiasAdd*
T0*'
_output_shapes
:?????????
?
/dense_4/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_4/kernel*
valueB"       *
dtype0*
_output_shapes
:
?
-dense_4/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_4/kernel*
valueB
 *???*
dtype0*
_output_shapes
: 
?
-dense_4/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_4/kernel*
valueB
 *??>*
dtype0*
_output_shapes
: 
?
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*
seed?*
T0*!
_class
loc:@dense_4/kernel*
dtype0*
seed2?*
_output_shapes

: 
?
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
: 
?
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

: 
?
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

: 
?
dense_4/kernel
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_4/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_4/kernel/AssignAssigndense_4/kernel)dense_4/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

: 
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

: 
?
dense_4/bias/Initializer/zerosConst*
_class
loc:@dense_4/bias*
valueB *    *
dtype0*
_output_shapes
: 
?
dense_4/bias
VariableV2*
shape: *
shared_name *
_class
loc:@dense_4/bias*
dtype0*
	container *
_output_shapes
: 
?
dense_4/bias/AssignAssigndense_4/biasdense_4/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
: 
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
: 
?
label2_output/dense_1/MatMulMatMullabel2_output/dense/Reludense_4/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
?
label2_output/dense_1/BiasAddBiasAddlabel2_output/dense_1/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:????????? 
s
label2_output/dense_1/ReluRelulabel2_output/dense_1/BiasAdd*
T0*'
_output_shapes
:????????? 
?
/dense_5/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_5/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
-dense_5/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_5/kernel*
valueB
 *0?*
dtype0*
_output_shapes
: 
?
-dense_5/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_5/kernel*
valueB
 *0?*
dtype0*
_output_shapes
: 
?
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
seed?*
T0*!
_class
loc:@dense_5/kernel*
dtype0*
seed2?*
_output_shapes

:
?
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 
?
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
?
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
?
dense_5/kernel
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_5/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_5/kernel/AssignAssigndense_5/kernel)dense_5/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
{
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
?
dense_5/bias/Initializer/zerosConst*
_class
loc:@dense_5/bias*
valueB*    *
dtype0*
_output_shapes
:
?
dense_5/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense_5/bias*
dtype0*
	container *
_output_shapes
:
?
dense_5/bias/AssignAssigndense_5/biasdense_5/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
q
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:
?
label2_output/dense_2/MatMulMatMullabel2_output/dense/Reludense_5/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
?
label2_output/dense_2/BiasAddBiasAddlabel2_output/dense_2/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
e
	Sigmoid_1Sigmoidlabel2_output/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
P
cvr_ctrIdentity	Sigmoid_1*
T0*'
_output_shapes
:?????????
x
loss/log_loss/CastCastPlaceholder*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:?????????
X
loss/log_loss/add/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
s
loss/log_loss/addAddloss/log_loss/Castloss/log_loss/add/y*
T0*'
_output_shapes
:?????????
]
loss/log_loss/LogLogloss/log_loss/add*
T0*'
_output_shapes
:?????????
b
loss/log_loss/MulMulctrloss/log_loss/Log*
T0*'
_output_shapes
:?????????
]
loss/log_loss/NegNegloss/log_loss/Mul*
T0*'
_output_shapes
:?????????
X
loss/log_loss/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
d
loss/log_loss/subSubloss/log_loss/sub/xctr*
T0*'
_output_shapes
:?????????
Z
loss/log_loss/sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
w
loss/log_loss/sub_1Subloss/log_loss/sub_1/xloss/log_loss/Cast*
T0*'
_output_shapes
:?????????
Z
loss/log_loss/add_1/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
x
loss/log_loss/add_1Addloss/log_loss/sub_1loss/log_loss/add_1/y*
T0*'
_output_shapes
:?????????
a
loss/log_loss/Log_1Logloss/log_loss/add_1*
T0*'
_output_shapes
:?????????
t
loss/log_loss/Mul_1Mulloss/log_loss/subloss/log_loss/Log_1*
T0*'
_output_shapes
:?????????
t
loss/log_loss/sub_2Subloss/log_loss/Negloss/log_loss/Mul_1*
T0*'
_output_shapes
:?????????
o
*loss/log_loss/assert_broadcastable/weightsConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
s
0loss/log_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
q
/loss/log_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
?
/loss/log_loss/assert_broadcastable/values/shapeShapeloss/log_loss/sub_2*
T0*
out_type0*
_output_shapes
:
p
.loss/log_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
F
>loss/log_loss/assert_broadcastable/static_scalar_check_successNoOp
?
loss/log_loss/Cast_1/xConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
y
loss/log_loss/Mul_2Mulloss/log_loss/sub_2loss/log_loss/Cast_1/x*
T0*'
_output_shapes
:?????????
?
loss/log_loss/ConstConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
?
loss/log_loss/SumSumloss/log_loss/Mul_2loss/log_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
!loss/log_loss/num_present/Equal/yConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/log_loss/num_present/EqualEqualloss/log_loss/Cast_1/x!loss/log_loss/num_present/Equal/y*
T0*
_output_shapes
: 
?
$loss/log_loss/num_present/zeros_likeConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
)loss/log_loss/num_present/ones_like/ShapeConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
)loss/log_loss/num_present/ones_like/ConstConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
#loss/log_loss/num_present/ones_likeFill)loss/log_loss/num_present/ones_like/Shape)loss/log_loss/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
?
 loss/log_loss/num_present/SelectSelectloss/log_loss/num_present/Equal$loss/log_loss/num_present/zeros_like#loss/log_loss/num_present/ones_like*
T0*
_output_shapes
: 
?
Nloss/log_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
Mloss/log_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
?
Mloss/log_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShapeloss/log_loss/sub_2?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
?
Lloss/log_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
?
\loss/log_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp?^loss/log_loss/assert_broadcastable/static_scalar_check_success
?
;loss/log_loss/num_present/broadcast_weights/ones_like/ShapeShapeloss/log_loss/sub_2?^loss/log_loss/assert_broadcastable/static_scalar_check_success]^loss/log_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
?
;loss/log_loss/num_present/broadcast_weights/ones_like/ConstConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success]^loss/log_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
5loss/log_loss/num_present/broadcast_weights/ones_likeFill;loss/log_loss/num_present/broadcast_weights/ones_like/Shape;loss/log_loss/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:?????????
?
+loss/log_loss/num_present/broadcast_weightsMul loss/log_loss/num_present/Select5loss/log_loss/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:?????????
?
loss/log_loss/num_present/ConstConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
?
loss/log_loss/num_presentSum+loss/log_loss/num_present/broadcast_weightsloss/log_loss/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
loss/log_loss/Const_1Const?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
loss/log_loss/Sum_1Sumloss/log_loss/Sumloss/log_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
p
loss/log_loss/valueDivNoNanloss/log_loss/Sum_1loss/log_loss/num_present*
T0*
_output_shapes
: 
M

loss/ConstConst*
valueB *
dtype0*
_output_shapes
: 
p
	loss/MeanMeanloss/log_loss/value
loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
z
loss/log_loss_1/CastCastPlaceholder*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:?????????
Z
loss/log_loss_1/add/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
y
loss/log_loss_1/addAddloss/log_loss_1/Castloss/log_loss_1/add/y*
T0*'
_output_shapes
:?????????
a
loss/log_loss_1/LogLogloss/log_loss_1/add*
T0*'
_output_shapes
:?????????
j
loss/log_loss_1/MulMulcvr_ctrloss/log_loss_1/Log*
T0*'
_output_shapes
:?????????
a
loss/log_loss_1/NegNegloss/log_loss_1/Mul*
T0*'
_output_shapes
:?????????
Z
loss/log_loss_1/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
l
loss/log_loss_1/subSubloss/log_loss_1/sub/xcvr_ctr*
T0*'
_output_shapes
:?????????
\
loss/log_loss_1/sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
}
loss/log_loss_1/sub_1Subloss/log_loss_1/sub_1/xloss/log_loss_1/Cast*
T0*'
_output_shapes
:?????????
\
loss/log_loss_1/add_1/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
~
loss/log_loss_1/add_1Addloss/log_loss_1/sub_1loss/log_loss_1/add_1/y*
T0*'
_output_shapes
:?????????
e
loss/log_loss_1/Log_1Logloss/log_loss_1/add_1*
T0*'
_output_shapes
:?????????
z
loss/log_loss_1/Mul_1Mulloss/log_loss_1/subloss/log_loss_1/Log_1*
T0*'
_output_shapes
:?????????
z
loss/log_loss_1/sub_2Subloss/log_loss_1/Negloss/log_loss_1/Mul_1*
T0*'
_output_shapes
:?????????
q
,loss/log_loss_1/assert_broadcastable/weightsConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
2loss/log_loss_1/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
s
1loss/log_loss_1/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
?
1loss/log_loss_1/assert_broadcastable/values/shapeShapeloss/log_loss_1/sub_2*
T0*
out_type0*
_output_shapes
:
r
0loss/log_loss_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
H
@loss/log_loss_1/assert_broadcastable/static_scalar_check_successNoOp
?
loss/log_loss_1/Cast_1/xConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 

loss/log_loss_1/Mul_2Mulloss/log_loss_1/sub_2loss/log_loss_1/Cast_1/x*
T0*'
_output_shapes
:?????????
?
loss/log_loss_1/ConstConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
?
loss/log_loss_1/SumSumloss/log_loss_1/Mul_2loss/log_loss_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
#loss/log_loss_1/num_present/Equal/yConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
!loss/log_loss_1/num_present/EqualEqualloss/log_loss_1/Cast_1/x#loss/log_loss_1/num_present/Equal/y*
T0*
_output_shapes
: 
?
&loss/log_loss_1/num_present/zeros_likeConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
+loss/log_loss_1/num_present/ones_like/ShapeConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
+loss/log_loss_1/num_present/ones_like/ConstConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
%loss/log_loss_1/num_present/ones_likeFill+loss/log_loss_1/num_present/ones_like/Shape+loss/log_loss_1/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
?
"loss/log_loss_1/num_present/SelectSelect!loss/log_loss_1/num_present/Equal&loss/log_loss_1/num_present/zeros_like%loss/log_loss_1/num_present/ones_like*
T0*
_output_shapes
: 
?
Ploss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
Oloss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/weights/rankConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
?
Oloss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/values/shapeShapeloss/log_loss_1/sub_2A^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
?
Nloss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/values/rankConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
?
^loss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success
?
=loss/log_loss_1/num_present/broadcast_weights/ones_like/ShapeShapeloss/log_loss_1/sub_2A^loss/log_loss_1/assert_broadcastable/static_scalar_check_success_^loss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
?
=loss/log_loss_1/num_present/broadcast_weights/ones_like/ConstConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success_^loss/log_loss_1/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7loss/log_loss_1/num_present/broadcast_weights/ones_likeFill=loss/log_loss_1/num_present/broadcast_weights/ones_like/Shape=loss/log_loss_1/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:?????????
?
-loss/log_loss_1/num_present/broadcast_weightsMul"loss/log_loss_1/num_present/Select7loss/log_loss_1/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:?????????
?
!loss/log_loss_1/num_present/ConstConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
?
loss/log_loss_1/num_presentSum-loss/log_loss_1/num_present/broadcast_weights!loss/log_loss_1/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
loss/log_loss_1/Const_1ConstA^loss/log_loss_1/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
loss/log_loss_1/Sum_1Sumloss/log_loss_1/Sumloss/log_loss_1/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
v
loss/log_loss_1/valueDivNoNanloss/log_loss_1/Sum_1loss/log_loss_1/num_present*
T0*
_output_shapes
: 
O
loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
v
loss/Mean_1Meanloss/log_loss_1/valueloss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
H
loss/addAdd	loss/Meanloss/Mean_1*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
A
(gradients/loss/add_grad/tuple/group_depsNoOp^gradients/Fill
?
0gradients/loss/add_grad/tuple/control_dependencyIdentitygradients/Fill)^gradients/loss/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
?
2gradients/loss/add_grad/tuple/control_dependency_1Identitygradients/Fill)^gradients/loss/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
i
&gradients/loss/Mean_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
 gradients/loss/Mean_grad/ReshapeReshape0gradients/loss/add_grad/tuple/control_dependency&gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
a
gradients/loss/Mean_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
gradients/loss/Mean_grad/TileTile gradients/loss/Mean_grad/Reshapegradients/loss/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
e
 gradients/loss/Mean_grad/Const_1Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 gradients/loss/Mean_grad/truedivRealDivgradients/loss/Mean_grad/Tile gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: 
k
(gradients/loss/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
"gradients/loss/Mean_1_grad/ReshapeReshape2gradients/loss/add_grad/tuple/control_dependency_1(gradients/loss/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
c
 gradients/loss/Mean_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
gradients/loss/Mean_1_grad/TileTile"gradients/loss/Mean_1_grad/Reshape gradients/loss/Mean_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
g
"gradients/loss/Mean_1_grad/Const_1Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
"gradients/loss/Mean_1_grad/truedivRealDivgradients/loss/Mean_1_grad/Tile"gradients/loss/Mean_1_grad/Const_1*
T0*
_output_shapes
: 
k
(gradients/loss/log_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
m
*gradients/loss/log_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
8gradients/loss/log_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/value_grad/Shape*gradients/loss/log_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
-gradients/loss/log_loss/value_grad/div_no_nanDivNoNan gradients/loss/Mean_grad/truedivloss/log_loss/num_present*
T0*
_output_shapes
: 
?
&gradients/loss/log_loss/value_grad/SumSum-gradients/loss/log_loss/value_grad/div_no_nan8gradients/loss/log_loss/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
*gradients/loss/log_loss/value_grad/ReshapeReshape&gradients/loss/log_loss/value_grad/Sum(gradients/loss/log_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
c
&gradients/loss/log_loss/value_grad/NegNegloss/log_loss/Sum_1*
T0*
_output_shapes
: 
?
/gradients/loss/log_loss/value_grad/div_no_nan_1DivNoNan&gradients/loss/log_loss/value_grad/Negloss/log_loss/num_present*
T0*
_output_shapes
: 
?
/gradients/loss/log_loss/value_grad/div_no_nan_2DivNoNan/gradients/loss/log_loss/value_grad/div_no_nan_1loss/log_loss/num_present*
T0*
_output_shapes
: 
?
&gradients/loss/log_loss/value_grad/mulMul gradients/loss/Mean_grad/truediv/gradients/loss/log_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
?
(gradients/loss/log_loss/value_grad/Sum_1Sum&gradients/loss/log_loss/value_grad/mul:gradients/loss/log_loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
,gradients/loss/log_loss/value_grad/Reshape_1Reshape(gradients/loss/log_loss/value_grad/Sum_1*gradients/loss/log_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
3gradients/loss/log_loss/value_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/value_grad/Reshape-^gradients/loss/log_loss/value_grad/Reshape_1
?
;gradients/loss/log_loss/value_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/value_grad/Reshape4^gradients/loss/log_loss/value_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/value_grad/Reshape*
_output_shapes
: 
?
=gradients/loss/log_loss/value_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/value_grad/Reshape_14^gradients/loss/log_loss/value_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/value_grad/Reshape_1*
_output_shapes
: 
m
*gradients/loss/log_loss_1/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
,gradients/loss/log_loss_1/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients/loss/log_loss_1/value_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/loss/log_loss_1/value_grad/Shape,gradients/loss/log_loss_1/value_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients/loss/log_loss_1/value_grad/div_no_nanDivNoNan"gradients/loss/Mean_1_grad/truedivloss/log_loss_1/num_present*
T0*
_output_shapes
: 
?
(gradients/loss/log_loss_1/value_grad/SumSum/gradients/loss/log_loss_1/value_grad/div_no_nan:gradients/loss/log_loss_1/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
,gradients/loss/log_loss_1/value_grad/ReshapeReshape(gradients/loss/log_loss_1/value_grad/Sum*gradients/loss/log_loss_1/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
g
(gradients/loss/log_loss_1/value_grad/NegNegloss/log_loss_1/Sum_1*
T0*
_output_shapes
: 
?
1gradients/loss/log_loss_1/value_grad/div_no_nan_1DivNoNan(gradients/loss/log_loss_1/value_grad/Negloss/log_loss_1/num_present*
T0*
_output_shapes
: 
?
1gradients/loss/log_loss_1/value_grad/div_no_nan_2DivNoNan1gradients/loss/log_loss_1/value_grad/div_no_nan_1loss/log_loss_1/num_present*
T0*
_output_shapes
: 
?
(gradients/loss/log_loss_1/value_grad/mulMul"gradients/loss/Mean_1_grad/truediv1gradients/loss/log_loss_1/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
?
*gradients/loss/log_loss_1/value_grad/Sum_1Sum(gradients/loss/log_loss_1/value_grad/mul<gradients/loss/log_loss_1/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
.gradients/loss/log_loss_1/value_grad/Reshape_1Reshape*gradients/loss/log_loss_1/value_grad/Sum_1,gradients/loss/log_loss_1/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients/loss/log_loss_1/value_grad/tuple/group_depsNoOp-^gradients/loss/log_loss_1/value_grad/Reshape/^gradients/loss/log_loss_1/value_grad/Reshape_1
?
=gradients/loss/log_loss_1/value_grad/tuple/control_dependencyIdentity,gradients/loss/log_loss_1/value_grad/Reshape6^gradients/loss/log_loss_1/value_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/value_grad/Reshape*
_output_shapes
: 
?
?gradients/loss/log_loss_1/value_grad/tuple/control_dependency_1Identity.gradients/loss/log_loss_1/value_grad/Reshape_16^gradients/loss/log_loss_1/value_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/loss/log_loss_1/value_grad/Reshape_1*
_output_shapes
: 
s
0gradients/loss/log_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
*gradients/loss/log_loss/Sum_1_grad/ReshapeReshape;gradients/loss/log_loss/value_grad/tuple/control_dependency0gradients/loss/log_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
k
(gradients/loss/log_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
'gradients/loss/log_loss/Sum_1_grad/TileTile*gradients/loss/log_loss/Sum_1_grad/Reshape(gradients/loss/log_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
u
2gradients/loss/log_loss_1/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
,gradients/loss/log_loss_1/Sum_1_grad/ReshapeReshape=gradients/loss/log_loss_1/value_grad/tuple/control_dependency2gradients/loss/log_loss_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
m
*gradients/loss/log_loss_1/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
)gradients/loss/log_loss_1/Sum_1_grad/TileTile,gradients/loss/log_loss_1/Sum_1_grad/Reshape*gradients/loss/log_loss_1/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

.gradients/loss/log_loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
(gradients/loss/log_loss/Sum_grad/ReshapeReshape'gradients/loss/log_loss/Sum_1_grad/Tile.gradients/loss/log_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
y
&gradients/loss/log_loss/Sum_grad/ShapeShapeloss/log_loss/Mul_2*
T0*
out_type0*
_output_shapes
:
?
%gradients/loss/log_loss/Sum_grad/TileTile(gradients/loss/log_loss/Sum_grad/Reshape&gradients/loss/log_loss/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:?????????
?
0gradients/loss/log_loss_1/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
*gradients/loss/log_loss_1/Sum_grad/ReshapeReshape)gradients/loss/log_loss_1/Sum_1_grad/Tile0gradients/loss/log_loss_1/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
}
(gradients/loss/log_loss_1/Sum_grad/ShapeShapeloss/log_loss_1/Mul_2*
T0*
out_type0*
_output_shapes
:
?
'gradients/loss/log_loss_1/Sum_grad/TileTile*gradients/loss/log_loss_1/Sum_grad/Reshape(gradients/loss/log_loss_1/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:?????????
{
(gradients/loss/log_loss/Mul_2_grad/ShapeShapeloss/log_loss/sub_2*
T0*
out_type0*
_output_shapes
:
m
*gradients/loss/log_loss/Mul_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
8gradients/loss/log_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/Mul_2_grad/Shape*gradients/loss/log_loss/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss/Mul_2_grad/MulMul%gradients/loss/log_loss/Sum_grad/Tileloss/log_loss/Cast_1/x*
T0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/Mul_2_grad/SumSum&gradients/loss/log_loss/Mul_2_grad/Mul8gradients/loss/log_loss/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/Mul_2_grad/ReshapeReshape&gradients/loss/log_loss/Mul_2_grad/Sum(gradients/loss/log_loss/Mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/Mul_2_grad/Mul_1Mulloss/log_loss/sub_2%gradients/loss/log_loss/Sum_grad/Tile*
T0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/Mul_2_grad/Sum_1Sum(gradients/loss/log_loss/Mul_2_grad/Mul_1:gradients/loss/log_loss/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss/Mul_2_grad/Reshape_1Reshape(gradients/loss/log_loss/Mul_2_grad/Sum_1*gradients/loss/log_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
3gradients/loss/log_loss/Mul_2_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/Mul_2_grad/Reshape-^gradients/loss/log_loss/Mul_2_grad/Reshape_1
?
;gradients/loss/log_loss/Mul_2_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/Mul_2_grad/Reshape4^gradients/loss/log_loss/Mul_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/Mul_2_grad/Reshape*'
_output_shapes
:?????????
?
=gradients/loss/log_loss/Mul_2_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/Mul_2_grad/Reshape_14^gradients/loss/log_loss/Mul_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/Mul_2_grad/Reshape_1*
_output_shapes
: 

*gradients/loss/log_loss_1/Mul_2_grad/ShapeShapeloss/log_loss_1/sub_2*
T0*
out_type0*
_output_shapes
:
o
,gradients/loss/log_loss_1/Mul_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients/loss/log_loss_1/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/loss/log_loss_1/Mul_2_grad/Shape,gradients/loss/log_loss_1/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients/loss/log_loss_1/Mul_2_grad/MulMul'gradients/loss/log_loss_1/Sum_grad/Tileloss/log_loss_1/Cast_1/x*
T0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss_1/Mul_2_grad/SumSum(gradients/loss/log_loss_1/Mul_2_grad/Mul:gradients/loss/log_loss_1/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss_1/Mul_2_grad/ReshapeReshape(gradients/loss/log_loss_1/Mul_2_grad/Sum*gradients/loss/log_loss_1/Mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients/loss/log_loss_1/Mul_2_grad/Mul_1Mulloss/log_loss_1/sub_2'gradients/loss/log_loss_1/Sum_grad/Tile*
T0*'
_output_shapes
:?????????
?
*gradients/loss/log_loss_1/Mul_2_grad/Sum_1Sum*gradients/loss/log_loss_1/Mul_2_grad/Mul_1<gradients/loss/log_loss_1/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients/loss/log_loss_1/Mul_2_grad/Reshape_1Reshape*gradients/loss/log_loss_1/Mul_2_grad/Sum_1,gradients/loss/log_loss_1/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients/loss/log_loss_1/Mul_2_grad/tuple/group_depsNoOp-^gradients/loss/log_loss_1/Mul_2_grad/Reshape/^gradients/loss/log_loss_1/Mul_2_grad/Reshape_1
?
=gradients/loss/log_loss_1/Mul_2_grad/tuple/control_dependencyIdentity,gradients/loss/log_loss_1/Mul_2_grad/Reshape6^gradients/loss/log_loss_1/Mul_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/Mul_2_grad/Reshape*'
_output_shapes
:?????????
?
?gradients/loss/log_loss_1/Mul_2_grad/tuple/control_dependency_1Identity.gradients/loss/log_loss_1/Mul_2_grad/Reshape_16^gradients/loss/log_loss_1/Mul_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/loss/log_loss_1/Mul_2_grad/Reshape_1*
_output_shapes
: 
y
(gradients/loss/log_loss/sub_2_grad/ShapeShapeloss/log_loss/Neg*
T0*
out_type0*
_output_shapes
:
}
*gradients/loss/log_loss/sub_2_grad/Shape_1Shapeloss/log_loss/Mul_1*
T0*
out_type0*
_output_shapes
:
?
8gradients/loss/log_loss/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/sub_2_grad/Shape*gradients/loss/log_loss/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss/sub_2_grad/SumSum;gradients/loss/log_loss/Mul_2_grad/tuple/control_dependency8gradients/loss/log_loss/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/sub_2_grad/ReshapeReshape&gradients/loss/log_loss/sub_2_grad/Sum(gradients/loss/log_loss/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/sub_2_grad/Sum_1Sum;gradients/loss/log_loss/Mul_2_grad/tuple/control_dependency:gradients/loss/log_loss/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
z
&gradients/loss/log_loss/sub_2_grad/NegNeg(gradients/loss/log_loss/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/loss/log_loss/sub_2_grad/Reshape_1Reshape&gradients/loss/log_loss/sub_2_grad/Neg*gradients/loss/log_loss/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/loss/log_loss/sub_2_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/sub_2_grad/Reshape-^gradients/loss/log_loss/sub_2_grad/Reshape_1
?
;gradients/loss/log_loss/sub_2_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/sub_2_grad/Reshape4^gradients/loss/log_loss/sub_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/sub_2_grad/Reshape*'
_output_shapes
:?????????
?
=gradients/loss/log_loss/sub_2_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/sub_2_grad/Reshape_14^gradients/loss/log_loss/sub_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/sub_2_grad/Reshape_1*'
_output_shapes
:?????????
}
*gradients/loss/log_loss_1/sub_2_grad/ShapeShapeloss/log_loss_1/Neg*
T0*
out_type0*
_output_shapes
:
?
,gradients/loss/log_loss_1/sub_2_grad/Shape_1Shapeloss/log_loss_1/Mul_1*
T0*
out_type0*
_output_shapes
:
?
:gradients/loss/log_loss_1/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/loss/log_loss_1/sub_2_grad/Shape,gradients/loss/log_loss_1/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients/loss/log_loss_1/sub_2_grad/SumSum=gradients/loss/log_loss_1/Mul_2_grad/tuple/control_dependency:gradients/loss/log_loss_1/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss_1/sub_2_grad/ReshapeReshape(gradients/loss/log_loss_1/sub_2_grad/Sum*gradients/loss/log_loss_1/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients/loss/log_loss_1/sub_2_grad/Sum_1Sum=gradients/loss/log_loss_1/Mul_2_grad/tuple/control_dependency<gradients/loss/log_loss_1/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
~
(gradients/loss/log_loss_1/sub_2_grad/NegNeg*gradients/loss/log_loss_1/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
.gradients/loss/log_loss_1/sub_2_grad/Reshape_1Reshape(gradients/loss/log_loss_1/sub_2_grad/Neg,gradients/loss/log_loss_1/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients/loss/log_loss_1/sub_2_grad/tuple/group_depsNoOp-^gradients/loss/log_loss_1/sub_2_grad/Reshape/^gradients/loss/log_loss_1/sub_2_grad/Reshape_1
?
=gradients/loss/log_loss_1/sub_2_grad/tuple/control_dependencyIdentity,gradients/loss/log_loss_1/sub_2_grad/Reshape6^gradients/loss/log_loss_1/sub_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/sub_2_grad/Reshape*'
_output_shapes
:?????????
?
?gradients/loss/log_loss_1/sub_2_grad/tuple/control_dependency_1Identity.gradients/loss/log_loss_1/sub_2_grad/Reshape_16^gradients/loss/log_loss_1/sub_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/loss/log_loss_1/sub_2_grad/Reshape_1*'
_output_shapes
:?????????
?
$gradients/loss/log_loss/Neg_grad/NegNeg;gradients/loss/log_loss/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
y
(gradients/loss/log_loss/Mul_1_grad/ShapeShapeloss/log_loss/sub*
T0*
out_type0*
_output_shapes
:
}
*gradients/loss/log_loss/Mul_1_grad/Shape_1Shapeloss/log_loss/Log_1*
T0*
out_type0*
_output_shapes
:
?
8gradients/loss/log_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/Mul_1_grad/Shape*gradients/loss/log_loss/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss/Mul_1_grad/MulMul=gradients/loss/log_loss/sub_2_grad/tuple/control_dependency_1loss/log_loss/Log_1*
T0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/Mul_1_grad/SumSum&gradients/loss/log_loss/Mul_1_grad/Mul8gradients/loss/log_loss/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/Mul_1_grad/ReshapeReshape&gradients/loss/log_loss/Mul_1_grad/Sum(gradients/loss/log_loss/Mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/Mul_1_grad/Mul_1Mulloss/log_loss/sub=gradients/loss/log_loss/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/Mul_1_grad/Sum_1Sum(gradients/loss/log_loss/Mul_1_grad/Mul_1:gradients/loss/log_loss/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss/Mul_1_grad/Reshape_1Reshape(gradients/loss/log_loss/Mul_1_grad/Sum_1*gradients/loss/log_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/loss/log_loss/Mul_1_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/Mul_1_grad/Reshape-^gradients/loss/log_loss/Mul_1_grad/Reshape_1
?
;gradients/loss/log_loss/Mul_1_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/Mul_1_grad/Reshape4^gradients/loss/log_loss/Mul_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/Mul_1_grad/Reshape*'
_output_shapes
:?????????
?
=gradients/loss/log_loss/Mul_1_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/Mul_1_grad/Reshape_14^gradients/loss/log_loss/Mul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/Mul_1_grad/Reshape_1*'
_output_shapes
:?????????
?
&gradients/loss/log_loss_1/Neg_grad/NegNeg=gradients/loss/log_loss_1/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
}
*gradients/loss/log_loss_1/Mul_1_grad/ShapeShapeloss/log_loss_1/sub*
T0*
out_type0*
_output_shapes
:
?
,gradients/loss/log_loss_1/Mul_1_grad/Shape_1Shapeloss/log_loss_1/Log_1*
T0*
out_type0*
_output_shapes
:
?
:gradients/loss/log_loss_1/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/loss/log_loss_1/Mul_1_grad/Shape,gradients/loss/log_loss_1/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients/loss/log_loss_1/Mul_1_grad/MulMul?gradients/loss/log_loss_1/sub_2_grad/tuple/control_dependency_1loss/log_loss_1/Log_1*
T0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss_1/Mul_1_grad/SumSum(gradients/loss/log_loss_1/Mul_1_grad/Mul:gradients/loss/log_loss_1/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss_1/Mul_1_grad/ReshapeReshape(gradients/loss/log_loss_1/Mul_1_grad/Sum*gradients/loss/log_loss_1/Mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients/loss/log_loss_1/Mul_1_grad/Mul_1Mulloss/log_loss_1/sub?gradients/loss/log_loss_1/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
*gradients/loss/log_loss_1/Mul_1_grad/Sum_1Sum*gradients/loss/log_loss_1/Mul_1_grad/Mul_1<gradients/loss/log_loss_1/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients/loss/log_loss_1/Mul_1_grad/Reshape_1Reshape*gradients/loss/log_loss_1/Mul_1_grad/Sum_1,gradients/loss/log_loss_1/Mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients/loss/log_loss_1/Mul_1_grad/tuple/group_depsNoOp-^gradients/loss/log_loss_1/Mul_1_grad/Reshape/^gradients/loss/log_loss_1/Mul_1_grad/Reshape_1
?
=gradients/loss/log_loss_1/Mul_1_grad/tuple/control_dependencyIdentity,gradients/loss/log_loss_1/Mul_1_grad/Reshape6^gradients/loss/log_loss_1/Mul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/Mul_1_grad/Reshape*'
_output_shapes
:?????????
?
?gradients/loss/log_loss_1/Mul_1_grad/tuple/control_dependency_1Identity.gradients/loss/log_loss_1/Mul_1_grad/Reshape_16^gradients/loss/log_loss_1/Mul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/loss/log_loss_1/Mul_1_grad/Reshape_1*'
_output_shapes
:?????????
i
&gradients/loss/log_loss/Mul_grad/ShapeShapectr*
T0*
out_type0*
_output_shapes
:
y
(gradients/loss/log_loss/Mul_grad/Shape_1Shapeloss/log_loss/Log*
T0*
out_type0*
_output_shapes
:
?
6gradients/loss/log_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/loss/log_loss/Mul_grad/Shape(gradients/loss/log_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/loss/log_loss/Mul_grad/MulMul$gradients/loss/log_loss/Neg_grad/Negloss/log_loss/Log*
T0*'
_output_shapes
:?????????
?
$gradients/loss/log_loss/Mul_grad/SumSum$gradients/loss/log_loss/Mul_grad/Mul6gradients/loss/log_loss/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients/loss/log_loss/Mul_grad/ReshapeReshape$gradients/loss/log_loss/Mul_grad/Sum&gradients/loss/log_loss/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/Mul_grad/Mul_1Mulctr$gradients/loss/log_loss/Neg_grad/Neg*
T0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/Mul_grad/Sum_1Sum&gradients/loss/log_loss/Mul_grad/Mul_18gradients/loss/log_loss/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/Mul_grad/Reshape_1Reshape&gradients/loss/log_loss/Mul_grad/Sum_1(gradients/loss/log_loss/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
1gradients/loss/log_loss/Mul_grad/tuple/group_depsNoOp)^gradients/loss/log_loss/Mul_grad/Reshape+^gradients/loss/log_loss/Mul_grad/Reshape_1
?
9gradients/loss/log_loss/Mul_grad/tuple/control_dependencyIdentity(gradients/loss/log_loss/Mul_grad/Reshape2^gradients/loss/log_loss/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/loss/log_loss/Mul_grad/Reshape*'
_output_shapes
:?????????
?
;gradients/loss/log_loss/Mul_grad/tuple/control_dependency_1Identity*gradients/loss/log_loss/Mul_grad/Reshape_12^gradients/loss/log_loss/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/Mul_grad/Reshape_1*'
_output_shapes
:?????????
i
&gradients/loss/log_loss/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
k
(gradients/loss/log_loss/sub_grad/Shape_1Shapectr*
T0*
out_type0*
_output_shapes
:
?
6gradients/loss/log_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/loss/log_loss/sub_grad/Shape(gradients/loss/log_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/loss/log_loss/sub_grad/SumSum;gradients/loss/log_loss/Mul_1_grad/tuple/control_dependency6gradients/loss/log_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients/loss/log_loss/sub_grad/ReshapeReshape$gradients/loss/log_loss/sub_grad/Sum&gradients/loss/log_loss/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
&gradients/loss/log_loss/sub_grad/Sum_1Sum;gradients/loss/log_loss/Mul_1_grad/tuple/control_dependency8gradients/loss/log_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
v
$gradients/loss/log_loss/sub_grad/NegNeg&gradients/loss/log_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
?
*gradients/loss/log_loss/sub_grad/Reshape_1Reshape$gradients/loss/log_loss/sub_grad/Neg(gradients/loss/log_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
1gradients/loss/log_loss/sub_grad/tuple/group_depsNoOp)^gradients/loss/log_loss/sub_grad/Reshape+^gradients/loss/log_loss/sub_grad/Reshape_1
?
9gradients/loss/log_loss/sub_grad/tuple/control_dependencyIdentity(gradients/loss/log_loss/sub_grad/Reshape2^gradients/loss/log_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/loss/log_loss/sub_grad/Reshape*
_output_shapes
: 
?
;gradients/loss/log_loss/sub_grad/tuple/control_dependency_1Identity*gradients/loss/log_loss/sub_grad/Reshape_12^gradients/loss/log_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/sub_grad/Reshape_1*'
_output_shapes
:?????????
o
(gradients/loss/log_loss_1/Mul_grad/ShapeShapecvr_ctr*
T0*
out_type0*
_output_shapes
:
}
*gradients/loss/log_loss_1/Mul_grad/Shape_1Shapeloss/log_loss_1/Log*
T0*
out_type0*
_output_shapes
:
?
8gradients/loss/log_loss_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss_1/Mul_grad/Shape*gradients/loss/log_loss_1/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss_1/Mul_grad/MulMul&gradients/loss/log_loss_1/Neg_grad/Negloss/log_loss_1/Log*
T0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss_1/Mul_grad/SumSum&gradients/loss/log_loss_1/Mul_grad/Mul8gradients/loss/log_loss_1/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss_1/Mul_grad/ReshapeReshape&gradients/loss/log_loss_1/Mul_grad/Sum(gradients/loss/log_loss_1/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss_1/Mul_grad/Mul_1Mulcvr_ctr&gradients/loss/log_loss_1/Neg_grad/Neg*
T0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss_1/Mul_grad/Sum_1Sum(gradients/loss/log_loss_1/Mul_grad/Mul_1:gradients/loss/log_loss_1/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss_1/Mul_grad/Reshape_1Reshape(gradients/loss/log_loss_1/Mul_grad/Sum_1*gradients/loss/log_loss_1/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/loss/log_loss_1/Mul_grad/tuple/group_depsNoOp+^gradients/loss/log_loss_1/Mul_grad/Reshape-^gradients/loss/log_loss_1/Mul_grad/Reshape_1
?
;gradients/loss/log_loss_1/Mul_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss_1/Mul_grad/Reshape4^gradients/loss/log_loss_1/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss_1/Mul_grad/Reshape*'
_output_shapes
:?????????
?
=gradients/loss/log_loss_1/Mul_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss_1/Mul_grad/Reshape_14^gradients/loss/log_loss_1/Mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/Mul_grad/Reshape_1*'
_output_shapes
:?????????
k
(gradients/loss/log_loss_1/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
q
*gradients/loss/log_loss_1/sub_grad/Shape_1Shapecvr_ctr*
T0*
out_type0*
_output_shapes
:
?
8gradients/loss/log_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss_1/sub_grad/Shape*gradients/loss/log_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss_1/sub_grad/SumSum=gradients/loss/log_loss_1/Mul_1_grad/tuple/control_dependency8gradients/loss/log_loss_1/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss_1/sub_grad/ReshapeReshape&gradients/loss/log_loss_1/sub_grad/Sum(gradients/loss/log_loss_1/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
(gradients/loss/log_loss_1/sub_grad/Sum_1Sum=gradients/loss/log_loss_1/Mul_1_grad/tuple/control_dependency:gradients/loss/log_loss_1/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
z
&gradients/loss/log_loss_1/sub_grad/NegNeg(gradients/loss/log_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/loss/log_loss_1/sub_grad/Reshape_1Reshape&gradients/loss/log_loss_1/sub_grad/Neg*gradients/loss/log_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/loss/log_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/loss/log_loss_1/sub_grad/Reshape-^gradients/loss/log_loss_1/sub_grad/Reshape_1
?
;gradients/loss/log_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss_1/sub_grad/Reshape4^gradients/loss/log_loss_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss_1/sub_grad/Reshape*
_output_shapes
: 
?
=gradients/loss/log_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss_1/sub_grad/Reshape_14^gradients/loss/log_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss_1/sub_grad/Reshape_1*'
_output_shapes
:?????????
?
gradients/AddNAddN9gradients/loss/log_loss/Mul_grad/tuple/control_dependency;gradients/loss/log_loss/sub_grad/tuple/control_dependency_1*
T0*;
_class1
/-loc:@gradients/loss/log_loss/Mul_grad/Reshape*
N*'
_output_shapes
:?????????
?
gradients/AddN_1AddN;gradients/loss/log_loss_1/Mul_grad/tuple/control_dependency=gradients/loss/log_loss_1/sub_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@gradients/loss/log_loss_1/Mul_grad/Reshape*
N*'
_output_shapes
:?????????
|
"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoidgradients/AddN*
T0*'
_output_shapes
:?????????
?
$gradients/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1gradients/AddN_1*
T0*'
_output_shapes
:?????????
?
8gradients/label1_output/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
=gradients/label1_output/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients/Sigmoid_grad/SigmoidGrad9^gradients/label1_output/dense_2/BiasAdd_grad/BiasAddGrad
?
Egradients/label1_output/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/Sigmoid_grad/SigmoidGrad>^gradients/label1_output/dense_2/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:?????????
?
Ggradients/label1_output/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/label1_output/dense_2/BiasAdd_grad/BiasAddGrad>^gradients/label1_output/dense_2/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/label1_output/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
8gradients/label2_output/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/Sigmoid_1_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
=gradients/label2_output/dense_2/BiasAdd_grad/tuple/group_depsNoOp%^gradients/Sigmoid_1_grad/SigmoidGrad9^gradients/label2_output/dense_2/BiasAdd_grad/BiasAddGrad
?
Egradients/label2_output/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/Sigmoid_1_grad/SigmoidGrad>^gradients/label2_output/dense_2/BiasAdd_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Sigmoid_1_grad/SigmoidGrad*'
_output_shapes
:?????????
?
Ggradients/label2_output/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/label2_output/dense_2/BiasAdd_grad/BiasAddGrad>^gradients/label2_output/dense_2/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/label2_output/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
2gradients/label1_output/dense_2/MatMul_grad/MatMulMatMulEgradients/label1_output/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:????????? 
?
4gradients/label1_output/dense_2/MatMul_grad/MatMul_1MatMullabel1_output/dense_1/ReluEgradients/label1_output/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

: 
?
<gradients/label1_output/dense_2/MatMul_grad/tuple/group_depsNoOp3^gradients/label1_output/dense_2/MatMul_grad/MatMul5^gradients/label1_output/dense_2/MatMul_grad/MatMul_1
?
Dgradients/label1_output/dense_2/MatMul_grad/tuple/control_dependencyIdentity2gradients/label1_output/dense_2/MatMul_grad/MatMul=^gradients/label1_output/dense_2/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label1_output/dense_2/MatMul_grad/MatMul*'
_output_shapes
:????????? 
?
Fgradients/label1_output/dense_2/MatMul_grad/tuple/control_dependency_1Identity4gradients/label1_output/dense_2/MatMul_grad/MatMul_1=^gradients/label1_output/dense_2/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/label1_output/dense_2/MatMul_grad/MatMul_1*
_output_shapes

: 
?
2gradients/label2_output/dense_2/MatMul_grad/MatMulMatMulEgradients/label2_output/dense_2/BiasAdd_grad/tuple/control_dependencydense_5/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
4gradients/label2_output/dense_2/MatMul_grad/MatMul_1MatMullabel2_output/dense/ReluEgradients/label2_output/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
?
<gradients/label2_output/dense_2/MatMul_grad/tuple/group_depsNoOp3^gradients/label2_output/dense_2/MatMul_grad/MatMul5^gradients/label2_output/dense_2/MatMul_grad/MatMul_1
?
Dgradients/label2_output/dense_2/MatMul_grad/tuple/control_dependencyIdentity2gradients/label2_output/dense_2/MatMul_grad/MatMul=^gradients/label2_output/dense_2/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label2_output/dense_2/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Fgradients/label2_output/dense_2/MatMul_grad/tuple/control_dependency_1Identity4gradients/label2_output/dense_2/MatMul_grad/MatMul_1=^gradients/label2_output/dense_2/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/label2_output/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:
?
2gradients/label1_output/dense_1/Relu_grad/ReluGradReluGradDgradients/label1_output/dense_2/MatMul_grad/tuple/control_dependencylabel1_output/dense_1/Relu*
T0*'
_output_shapes
:????????? 
?
0gradients/label2_output/dense/Relu_grad/ReluGradReluGradDgradients/label2_output/dense_2/MatMul_grad/tuple/control_dependencylabel2_output/dense/Relu*
T0*'
_output_shapes
:?????????
?
8gradients/label1_output/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/label1_output/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
?
=gradients/label1_output/dense_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients/label1_output/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/label1_output/dense_1/Relu_grad/ReluGrad
?
Egradients/label1_output/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/label1_output/dense_1/Relu_grad/ReluGrad>^gradients/label1_output/dense_1/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label1_output/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:????????? 
?
Ggradients/label1_output/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/label1_output/dense_1/BiasAdd_grad/BiasAddGrad>^gradients/label1_output/dense_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/label1_output/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
6gradients/label2_output/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/label2_output/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
;gradients/label2_output/dense/BiasAdd_grad/tuple/group_depsNoOp7^gradients/label2_output/dense/BiasAdd_grad/BiasAddGrad1^gradients/label2_output/dense/Relu_grad/ReluGrad
?
Cgradients/label2_output/dense/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/label2_output/dense/Relu_grad/ReluGrad<^gradients/label2_output/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/label2_output/dense/Relu_grad/ReluGrad*'
_output_shapes
:?????????
?
Egradients/label2_output/dense/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/label2_output/dense/BiasAdd_grad/BiasAddGrad<^gradients/label2_output/dense/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/label2_output/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
2gradients/label1_output/dense_1/MatMul_grad/MatMulMatMulEgradients/label1_output/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
4gradients/label1_output/dense_1/MatMul_grad/MatMul_1MatMullabel1_output/dense/ReluEgradients/label1_output/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

: 
?
<gradients/label1_output/dense_1/MatMul_grad/tuple/group_depsNoOp3^gradients/label1_output/dense_1/MatMul_grad/MatMul5^gradients/label1_output/dense_1/MatMul_grad/MatMul_1
?
Dgradients/label1_output/dense_1/MatMul_grad/tuple/control_dependencyIdentity2gradients/label1_output/dense_1/MatMul_grad/MatMul=^gradients/label1_output/dense_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label1_output/dense_1/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Fgradients/label1_output/dense_1/MatMul_grad/tuple/control_dependency_1Identity4gradients/label1_output/dense_1/MatMul_grad/MatMul_1=^gradients/label1_output/dense_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/label1_output/dense_1/MatMul_grad/MatMul_1*
_output_shapes

: 
?
0gradients/label2_output/dense/MatMul_grad/MatMulMatMulCgradients/label2_output/dense/BiasAdd_grad/tuple/control_dependencydense_3/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
2gradients/label2_output/dense/MatMul_grad/MatMul_1MatMullabel2_input/ReshapeCgradients/label2_output/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
?
:gradients/label2_output/dense/MatMul_grad/tuple/group_depsNoOp1^gradients/label2_output/dense/MatMul_grad/MatMul3^gradients/label2_output/dense/MatMul_grad/MatMul_1
?
Bgradients/label2_output/dense/MatMul_grad/tuple/control_dependencyIdentity0gradients/label2_output/dense/MatMul_grad/MatMul;^gradients/label2_output/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/label2_output/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Dgradients/label2_output/dense/MatMul_grad/tuple/control_dependency_1Identity2gradients/label2_output/dense/MatMul_grad/MatMul_1;^gradients/label2_output/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label2_output/dense/MatMul_grad/MatMul_1*
_output_shapes

:
?
0gradients/label1_output/dense/Relu_grad/ReluGradReluGradDgradients/label1_output/dense_1/MatMul_grad/tuple/control_dependencylabel1_output/dense/Relu*
T0*'
_output_shapes
:?????????
y
)gradients/label2_input/Reshape_grad/ShapeShapelabel2_input/Sum*
T0*
out_type0*
_output_shapes
:
?
+gradients/label2_input/Reshape_grad/ReshapeReshapeBgradients/label2_output/dense/MatMul_grad/tuple/control_dependency)gradients/label2_input/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6gradients/label1_output/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/label1_output/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
;gradients/label1_output/dense/BiasAdd_grad/tuple/group_depsNoOp7^gradients/label1_output/dense/BiasAdd_grad/BiasAddGrad1^gradients/label1_output/dense/Relu_grad/ReluGrad
?
Cgradients/label1_output/dense/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/label1_output/dense/Relu_grad/ReluGrad<^gradients/label1_output/dense/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/label1_output/dense/Relu_grad/ReluGrad*'
_output_shapes
:?????????
?
Egradients/label1_output/dense/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/label1_output/dense/BiasAdd_grad/BiasAddGrad<^gradients/label1_output/dense/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/label1_output/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
u
%gradients/label2_input/Sum_grad/ShapeShapelabel2_input/Mul*
T0*
out_type0*
_output_shapes
:
?
$gradients/label2_input/Sum_grad/SizeConst*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
#gradients/label2_input/Sum_grad/addAdd"label2_input/Sum/reduction_indices$gradients/label2_input/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
_output_shapes
: 
?
#gradients/label2_input/Sum_grad/modFloorMod#gradients/label2_input/Sum_grad/add$gradients/label2_input/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
_output_shapes
: 
?
'gradients/label2_input/Sum_grad/Shape_1Const*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
+gradients/label2_input/Sum_grad/range/startConst*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
+gradients/label2_input/Sum_grad/range/deltaConst*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients/label2_input/Sum_grad/rangeRange+gradients/label2_input/Sum_grad/range/start$gradients/label2_input/Sum_grad/Size+gradients/label2_input/Sum_grad/range/delta*

Tidx0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
_output_shapes
:
?
*gradients/label2_input/Sum_grad/Fill/valueConst*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients/label2_input/Sum_grad/FillFill'gradients/label2_input/Sum_grad/Shape_1*gradients/label2_input/Sum_grad/Fill/value*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
-gradients/label2_input/Sum_grad/DynamicStitchDynamicStitch%gradients/label2_input/Sum_grad/range#gradients/label2_input/Sum_grad/mod%gradients/label2_input/Sum_grad/Shape$gradients/label2_input/Sum_grad/Fill*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
N*
_output_shapes
:
?
)gradients/label2_input/Sum_grad/Maximum/yConst*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
'gradients/label2_input/Sum_grad/MaximumMaximum-gradients/label2_input/Sum_grad/DynamicStitch)gradients/label2_input/Sum_grad/Maximum/y*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
_output_shapes
:
?
(gradients/label2_input/Sum_grad/floordivFloorDiv%gradients/label2_input/Sum_grad/Shape'gradients/label2_input/Sum_grad/Maximum*
T0*8
_class.
,*loc:@gradients/label2_input/Sum_grad/Shape*
_output_shapes
:
?
'gradients/label2_input/Sum_grad/ReshapeReshape+gradients/label2_input/Reshape_grad/Reshape-gradients/label2_input/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
$gradients/label2_input/Sum_grad/TileTile'gradients/label2_input/Sum_grad/Reshape(gradients/label2_input/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
?
0gradients/label1_output/dense/MatMul_grad/MatMulMatMulCgradients/label1_output/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
2gradients/label1_output/dense/MatMul_grad/MatMul_1MatMullabel1_input/ReshapeCgradients/label1_output/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:
?
:gradients/label1_output/dense/MatMul_grad/tuple/group_depsNoOp1^gradients/label1_output/dense/MatMul_grad/MatMul3^gradients/label1_output/dense/MatMul_grad/MatMul_1
?
Bgradients/label1_output/dense/MatMul_grad/tuple/control_dependencyIdentity0gradients/label1_output/dense/MatMul_grad/MatMul;^gradients/label1_output/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/label1_output/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Dgradients/label1_output/dense/MatMul_grad/tuple/control_dependency_1Identity2gradients/label1_output/dense/MatMul_grad/MatMul_1;^gradients/label1_output/dense/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/label1_output/dense/MatMul_grad/MatMul_1*
_output_shapes

:
v
%gradients/label2_input/Mul_grad/ShapeShapeexpert/expert_out*
T0*
out_type0*
_output_shapes
:
~
'gradients/label2_input/Mul_grad/Shape_1Shapelabel2_input/ExpandDims*
T0*
out_type0*
_output_shapes
:
?
5gradients/label2_input/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/label2_input/Mul_grad/Shape'gradients/label2_input/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
#gradients/label2_input/Mul_grad/MulMul$gradients/label2_input/Sum_grad/Tilelabel2_input/ExpandDims*
T0*+
_output_shapes
:?????????
?
#gradients/label2_input/Mul_grad/SumSum#gradients/label2_input/Mul_grad/Mul5gradients/label2_input/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
'gradients/label2_input/Mul_grad/ReshapeReshape#gradients/label2_input/Mul_grad/Sum%gradients/label2_input/Mul_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
%gradients/label2_input/Mul_grad/Mul_1Mulexpert/expert_out$gradients/label2_input/Sum_grad/Tile*
T0*+
_output_shapes
:?????????
?
%gradients/label2_input/Mul_grad/Sum_1Sum%gradients/label2_input/Mul_grad/Mul_17gradients/label2_input/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
)gradients/label2_input/Mul_grad/Reshape_1Reshape%gradients/label2_input/Mul_grad/Sum_1'gradients/label2_input/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
0gradients/label2_input/Mul_grad/tuple/group_depsNoOp(^gradients/label2_input/Mul_grad/Reshape*^gradients/label2_input/Mul_grad/Reshape_1
?
8gradients/label2_input/Mul_grad/tuple/control_dependencyIdentity'gradients/label2_input/Mul_grad/Reshape1^gradients/label2_input/Mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/label2_input/Mul_grad/Reshape*+
_output_shapes
:?????????
?
:gradients/label2_input/Mul_grad/tuple/control_dependency_1Identity)gradients/label2_input/Mul_grad/Reshape_11^gradients/label2_input/Mul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/label2_input/Mul_grad/Reshape_1*+
_output_shapes
:?????????
y
)gradients/label1_input/Reshape_grad/ShapeShapelabel1_input/Sum*
T0*
out_type0*
_output_shapes
:
?
+gradients/label1_input/Reshape_grad/ReshapeReshapeBgradients/label1_output/dense/MatMul_grad/tuple/control_dependency)gradients/label1_input/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
{
,gradients/label2_input/ExpandDims_grad/ShapeShapegate2/gate2_out*
T0*
out_type0*
_output_shapes
:
?
.gradients/label2_input/ExpandDims_grad/ReshapeReshape:gradients/label2_input/Mul_grad/tuple/control_dependency_1,gradients/label2_input/ExpandDims_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
u
%gradients/label1_input/Sum_grad/ShapeShapelabel1_input/Mul*
T0*
out_type0*
_output_shapes
:
?
$gradients/label1_input/Sum_grad/SizeConst*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
#gradients/label1_input/Sum_grad/addAdd"label1_input/Sum/reduction_indices$gradients/label1_input/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
_output_shapes
: 
?
#gradients/label1_input/Sum_grad/modFloorMod#gradients/label1_input/Sum_grad/add$gradients/label1_input/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
_output_shapes
: 
?
'gradients/label1_input/Sum_grad/Shape_1Const*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
+gradients/label1_input/Sum_grad/range/startConst*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
+gradients/label1_input/Sum_grad/range/deltaConst*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients/label1_input/Sum_grad/rangeRange+gradients/label1_input/Sum_grad/range/start$gradients/label1_input/Sum_grad/Size+gradients/label1_input/Sum_grad/range/delta*

Tidx0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
_output_shapes
:
?
*gradients/label1_input/Sum_grad/Fill/valueConst*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients/label1_input/Sum_grad/FillFill'gradients/label1_input/Sum_grad/Shape_1*gradients/label1_input/Sum_grad/Fill/value*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
-gradients/label1_input/Sum_grad/DynamicStitchDynamicStitch%gradients/label1_input/Sum_grad/range#gradients/label1_input/Sum_grad/mod%gradients/label1_input/Sum_grad/Shape$gradients/label1_input/Sum_grad/Fill*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
N*
_output_shapes
:
?
)gradients/label1_input/Sum_grad/Maximum/yConst*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
'gradients/label1_input/Sum_grad/MaximumMaximum-gradients/label1_input/Sum_grad/DynamicStitch)gradients/label1_input/Sum_grad/Maximum/y*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
_output_shapes
:
?
(gradients/label1_input/Sum_grad/floordivFloorDiv%gradients/label1_input/Sum_grad/Shape'gradients/label1_input/Sum_grad/Maximum*
T0*8
_class.
,*loc:@gradients/label1_input/Sum_grad/Shape*
_output_shapes
:
?
'gradients/label1_input/Sum_grad/ReshapeReshape+gradients/label1_input/Reshape_grad/Reshape-gradients/label1_input/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
$gradients/label1_input/Sum_grad/TileTile'gradients/label1_input/Sum_grad/Reshape(gradients/label1_input/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
?
"gradients/gate2/gate2_out_grad/mulMul.gradients/label2_input/ExpandDims_grad/Reshapegate2/gate2_out*
T0*'
_output_shapes
:?????????

4gradients/gate2/gate2_out_grad/Sum/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
"gradients/gate2/gate2_out_grad/SumSum"gradients/gate2/gate2_out_grad/mul4gradients/gate2/gate2_out_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:?????????
?
"gradients/gate2/gate2_out_grad/subSub.gradients/label2_input/ExpandDims_grad/Reshape"gradients/gate2/gate2_out_grad/Sum*
T0*'
_output_shapes
:?????????
?
$gradients/gate2/gate2_out_grad/mul_1Mul"gradients/gate2/gate2_out_grad/subgate2/gate2_out*
T0*'
_output_shapes
:?????????
v
%gradients/label1_input/Mul_grad/ShapeShapeexpert/expert_out*
T0*
out_type0*
_output_shapes
:
~
'gradients/label1_input/Mul_grad/Shape_1Shapelabel1_input/ExpandDims*
T0*
out_type0*
_output_shapes
:
?
5gradients/label1_input/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/label1_input/Mul_grad/Shape'gradients/label1_input/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
#gradients/label1_input/Mul_grad/MulMul$gradients/label1_input/Sum_grad/Tilelabel1_input/ExpandDims*
T0*+
_output_shapes
:?????????
?
#gradients/label1_input/Mul_grad/SumSum#gradients/label1_input/Mul_grad/Mul5gradients/label1_input/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
'gradients/label1_input/Mul_grad/ReshapeReshape#gradients/label1_input/Mul_grad/Sum%gradients/label1_input/Mul_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
%gradients/label1_input/Mul_grad/Mul_1Mulexpert/expert_out$gradients/label1_input/Sum_grad/Tile*
T0*+
_output_shapes
:?????????
?
%gradients/label1_input/Mul_grad/Sum_1Sum%gradients/label1_input/Mul_grad/Mul_17gradients/label1_input/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
)gradients/label1_input/Mul_grad/Reshape_1Reshape%gradients/label1_input/Mul_grad/Sum_1'gradients/label1_input/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
0gradients/label1_input/Mul_grad/tuple/group_depsNoOp(^gradients/label1_input/Mul_grad/Reshape*^gradients/label1_input/Mul_grad/Reshape_1
?
8gradients/label1_input/Mul_grad/tuple/control_dependencyIdentity'gradients/label1_input/Mul_grad/Reshape1^gradients/label1_input/Mul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/label1_input/Mul_grad/Reshape*+
_output_shapes
:?????????
?
:gradients/label1_input/Mul_grad/tuple/control_dependency_1Identity)gradients/label1_input/Mul_grad/Reshape_11^gradients/label1_input/Mul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/label1_input/Mul_grad/Reshape_1*+
_output_shapes
:?????????
j
gradients/gate2/Add_grad/ShapeShapegate2/MatMul*
T0*
out_type0*
_output_shapes
:
j
 gradients/gate2/Add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
.gradients/gate2/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/gate2/Add_grad/Shape gradients/gate2/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/gate2/Add_grad/SumSum$gradients/gate2/gate2_out_grad/mul_1.gradients/gate2/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
 gradients/gate2/Add_grad/ReshapeReshapegradients/gate2/Add_grad/Sumgradients/gate2/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/gate2/Add_grad/Sum_1Sum$gradients/gate2/gate2_out_grad/mul_10gradients/gate2/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
"gradients/gate2/Add_grad/Reshape_1Reshapegradients/gate2/Add_grad/Sum_1 gradients/gate2/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
y
)gradients/gate2/Add_grad/tuple/group_depsNoOp!^gradients/gate2/Add_grad/Reshape#^gradients/gate2/Add_grad/Reshape_1
?
1gradients/gate2/Add_grad/tuple/control_dependencyIdentity gradients/gate2/Add_grad/Reshape*^gradients/gate2/Add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/gate2/Add_grad/Reshape*'
_output_shapes
:?????????
?
3gradients/gate2/Add_grad/tuple/control_dependency_1Identity"gradients/gate2/Add_grad/Reshape_1*^gradients/gate2/Add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/gate2/Add_grad/Reshape_1*
_output_shapes
:
?
gradients/AddN_2AddN8gradients/label2_input/Mul_grad/tuple/control_dependency8gradients/label1_input/Mul_grad/tuple/control_dependency*
T0*:
_class0
.,loc:@gradients/label2_input/Mul_grad/Reshape*
N*+
_output_shapes
:?????????
?
)gradients/expert/expert_out_grad/ReluGradReluGradgradients/AddN_2expert/expert_out*
T0*+
_output_shapes
:?????????
{
,gradients/label1_input/ExpandDims_grad/ShapeShapegate1/gate1_out*
T0*
out_type0*
_output_shapes
:
?
.gradients/label1_input/ExpandDims_grad/ReshapeReshape:gradients/label1_input/Mul_grad/tuple/control_dependency_1,gradients/label1_input/ExpandDims_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
"gradients/gate2/MatMul_grad/MatMulMatMul1gradients/gate2/Add_grad/tuple/control_dependencygate2_weight/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
$gradients/gate2/MatMul_grad/MatMul_1MatMulhidden/Reshape1gradients/gate2/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	?
?
,gradients/gate2/MatMul_grad/tuple/group_depsNoOp#^gradients/gate2/MatMul_grad/MatMul%^gradients/gate2/MatMul_grad/MatMul_1
?
4gradients/gate2/MatMul_grad/tuple/control_dependencyIdentity"gradients/gate2/MatMul_grad/MatMul-^gradients/gate2/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/gate2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
6gradients/gate2/MatMul_grad/tuple/control_dependency_1Identity$gradients/gate2/MatMul_grad/MatMul_1-^gradients/gate2/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/gate2/MatMul_grad/MatMul_1*
_output_shapes
:	?
o
gradients/expert/Add_grad/ShapeShapeexpert/Tensordot*
T0*
out_type0*
_output_shapes
:
k
!gradients/expert/Add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
/gradients/expert/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/expert/Add_grad/Shape!gradients/expert/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/expert/Add_grad/SumSum)gradients/expert/expert_out_grad/ReluGrad/gradients/expert/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
!gradients/expert/Add_grad/ReshapeReshapegradients/expert/Add_grad/Sumgradients/expert/Add_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
gradients/expert/Add_grad/Sum_1Sum)gradients/expert/expert_out_grad/ReluGrad1gradients/expert/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
#gradients/expert/Add_grad/Reshape_1Reshapegradients/expert/Add_grad/Sum_1!gradients/expert/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
|
*gradients/expert/Add_grad/tuple/group_depsNoOp"^gradients/expert/Add_grad/Reshape$^gradients/expert/Add_grad/Reshape_1
?
2gradients/expert/Add_grad/tuple/control_dependencyIdentity!gradients/expert/Add_grad/Reshape+^gradients/expert/Add_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/expert/Add_grad/Reshape*+
_output_shapes
:?????????
?
4gradients/expert/Add_grad/tuple/control_dependency_1Identity#gradients/expert/Add_grad/Reshape_1+^gradients/expert/Add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/expert/Add_grad/Reshape_1*
_output_shapes
:
?
"gradients/gate1/gate1_out_grad/mulMul.gradients/label1_input/ExpandDims_grad/Reshapegate1/gate1_out*
T0*'
_output_shapes
:?????????

4gradients/gate1/gate1_out_grad/Sum/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
"gradients/gate1/gate1_out_grad/SumSum"gradients/gate1/gate1_out_grad/mul4gradients/gate1/gate1_out_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:?????????
?
"gradients/gate1/gate1_out_grad/subSub.gradients/label1_input/ExpandDims_grad/Reshape"gradients/gate1/gate1_out_grad/Sum*
T0*'
_output_shapes
:?????????
?
$gradients/gate1/gate1_out_grad/mul_1Mul"gradients/gate1/gate1_out_grad/subgate1/gate1_out*
T0*'
_output_shapes
:?????????
|
%gradients/expert/Tensordot_grad/ShapeShapeexpert/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
'gradients/expert/Tensordot_grad/ReshapeReshape2gradients/expert/Add_grad/tuple/control_dependency%gradients/expert/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????@
j
gradients/gate1/Add_grad/ShapeShapegate1/MatMul*
T0*
out_type0*
_output_shapes
:
j
 gradients/gate1/Add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
.gradients/gate1/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/gate1/Add_grad/Shape gradients/gate1/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/gate1/Add_grad/SumSum$gradients/gate1/gate1_out_grad/mul_1.gradients/gate1/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
 gradients/gate1/Add_grad/ReshapeReshapegradients/gate1/Add_grad/Sumgradients/gate1/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/gate1/Add_grad/Sum_1Sum$gradients/gate1/gate1_out_grad/mul_10gradients/gate1/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
"gradients/gate1/Add_grad/Reshape_1Reshapegradients/gate1/Add_grad/Sum_1 gradients/gate1/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
y
)gradients/gate1/Add_grad/tuple/group_depsNoOp!^gradients/gate1/Add_grad/Reshape#^gradients/gate1/Add_grad/Reshape_1
?
1gradients/gate1/Add_grad/tuple/control_dependencyIdentity gradients/gate1/Add_grad/Reshape*^gradients/gate1/Add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/gate1/Add_grad/Reshape*'
_output_shapes
:?????????
?
3gradients/gate1/Add_grad/tuple/control_dependency_1Identity"gradients/gate1/Add_grad/Reshape_1*^gradients/gate1/Add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/gate1/Add_grad/Reshape_1*
_output_shapes
:
?
-gradients/expert/Tensordot/MatMul_grad/MatMulMatMul'gradients/expert/Tensordot_grad/Reshapeexpert/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
/gradients/expert/Tensordot/MatMul_grad/MatMul_1MatMulexpert/Tensordot/Reshape'gradients/expert/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????@
?
7gradients/expert/Tensordot/MatMul_grad/tuple/group_depsNoOp.^gradients/expert/Tensordot/MatMul_grad/MatMul0^gradients/expert/Tensordot/MatMul_grad/MatMul_1
?
?gradients/expert/Tensordot/MatMul_grad/tuple/control_dependencyIdentity-gradients/expert/Tensordot/MatMul_grad/MatMul8^gradients/expert/Tensordot/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/expert/Tensordot/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Agradients/expert/Tensordot/MatMul_grad/tuple/control_dependency_1Identity/gradients/expert/Tensordot/MatMul_grad/MatMul_18^gradients/expert/Tensordot/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/expert/Tensordot/MatMul_grad/MatMul_1*
_output_shapes
:	?@
?
"gradients/gate1/MatMul_grad/MatMulMatMul1gradients/gate1/Add_grad/tuple/control_dependencygate1_weight/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
$gradients/gate1/MatMul_grad/MatMul_1MatMulhidden/Reshape1gradients/gate1/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	?
?
,gradients/gate1/MatMul_grad/tuple/group_depsNoOp#^gradients/gate1/MatMul_grad/MatMul%^gradients/gate1/MatMul_grad/MatMul_1
?
4gradients/gate1/MatMul_grad/tuple/control_dependencyIdentity"gradients/gate1/MatMul_grad/MatMul-^gradients/gate1/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/gate1/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
6gradients/gate1/MatMul_grad/tuple/control_dependency_1Identity$gradients/gate1/MatMul_grad/MatMul_1-^gradients/gate1/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/gate1/MatMul_grad/MatMul_1*
_output_shapes
:	?
?
-gradients/expert/Tensordot/Reshape_grad/ShapeShapeexpert/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
/gradients/expert/Tensordot/Reshape_grad/ReshapeReshape?gradients/expert/Tensordot/MatMul_grad/tuple/control_dependency-gradients/expert/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
/gradients/expert/Tensordot/Reshape_1_grad/ShapeConst*!
valueB"?         *
dtype0*
_output_shapes
:
?
1gradients/expert/Tensordot/Reshape_1_grad/ReshapeReshapeAgradients/expert/Tensordot/MatMul_grad/tuple/control_dependency_1/gradients/expert/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?
?
;gradients/expert/Tensordot/transpose_grad/InvertPermutationInvertPermutationexpert/Tensordot/concat*
T0*
_output_shapes
:
?
3gradients/expert/Tensordot/transpose_grad/transpose	Transpose/gradients/expert/Tensordot/Reshape_grad/Reshape;gradients/expert/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*(
_output_shapes
:??????????
?
=gradients/expert/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation!expert/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
5gradients/expert/Tensordot/transpose_1_grad/transpose	Transpose1gradients/expert/Tensordot/Reshape_1_grad/Reshape=gradients/expert/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*#
_output_shapes
:?
?
gradients/AddN_3AddN4gradients/gate2/MatMul_grad/tuple/control_dependency4gradients/gate1/MatMul_grad/tuple/control_dependency3gradients/expert/Tensordot/transpose_grad/transpose*
T0*5
_class+
)'loc:@gradients/gate2/MatMul_grad/MatMul*
N*(
_output_shapes
:??????????
p
#gradients/hidden/Reshape_grad/ShapeShapehidden/concat*
T0*
out_type0*
_output_shapes
:
?
%gradients/hidden/Reshape_grad/ReshapeReshapegradients/AddN_3#gradients/hidden/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:??????????
c
!gradients/hidden/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
 gradients/hidden/concat_grad/modFloorModhidden/concat/axis!gradients/hidden/concat_grad/Rank*
T0*
_output_shapes
: 
|
"gradients/hidden/concat_grad/ShapeShapeuser_fc/u_concated_fc/Relu*
T0*
out_type0*
_output_shapes
:
?
#gradients/hidden/concat_grad/ShapeNShapeNuser_fc/u_concated_fc/Relu#item_concated_fc/i_concated_fc/Relu*
T0*
out_type0*
N* 
_output_shapes
::
?
)gradients/hidden/concat_grad/ConcatOffsetConcatOffset gradients/hidden/concat_grad/mod#gradients/hidden/concat_grad/ShapeN%gradients/hidden/concat_grad/ShapeN:1*
N* 
_output_shapes
::
?
"gradients/hidden/concat_grad/SliceSlice%gradients/hidden/Reshape_grad/Reshape)gradients/hidden/concat_grad/ConcatOffset#gradients/hidden/concat_grad/ShapeN*
T0*
Index0*+
_output_shapes
:?????????@
?
$gradients/hidden/concat_grad/Slice_1Slice%gradients/hidden/Reshape_grad/Reshape+gradients/hidden/concat_grad/ConcatOffset:1%gradients/hidden/concat_grad/ShapeN:1*
T0*
Index0*+
_output_shapes
:?????????@
?
-gradients/hidden/concat_grad/tuple/group_depsNoOp#^gradients/hidden/concat_grad/Slice%^gradients/hidden/concat_grad/Slice_1
?
5gradients/hidden/concat_grad/tuple/control_dependencyIdentity"gradients/hidden/concat_grad/Slice.^gradients/hidden/concat_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/hidden/concat_grad/Slice*+
_output_shapes
:?????????@
?
7gradients/hidden/concat_grad/tuple/control_dependency_1Identity$gradients/hidden/concat_grad/Slice_1.^gradients/hidden/concat_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/hidden/concat_grad/Slice_1*+
_output_shapes
:?????????@
?
2gradients/user_fc/u_concated_fc/Relu_grad/ReluGradReluGrad5gradients/hidden/concat_grad/tuple/control_dependencyuser_fc/u_concated_fc/Relu*
T0*+
_output_shapes
:?????????@
?
;gradients/item_concated_fc/i_concated_fc/Relu_grad/ReluGradReluGrad7gradients/hidden/concat_grad/tuple/control_dependency_1#item_concated_fc/i_concated_fc/Relu*
T0*+
_output_shapes
:?????????@
?
8gradients/user_fc/u_concated_fc/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/user_fc/u_concated_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
=gradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/group_depsNoOp9^gradients/user_fc/u_concated_fc/BiasAdd_grad/BiasAddGrad3^gradients/user_fc/u_concated_fc/Relu_grad/ReluGrad
?
Egradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/user_fc/u_concated_fc/Relu_grad/ReluGrad>^gradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/user_fc/u_concated_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????@
?
Ggradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/control_dependency_1Identity8gradients/user_fc/u_concated_fc/BiasAdd_grad/BiasAddGrad>^gradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/user_fc/u_concated_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
Agradients/item_concated_fc/i_concated_fc/BiasAdd_grad/BiasAddGradBiasAddGrad;gradients/item_concated_fc/i_concated_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
Fgradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/group_depsNoOpB^gradients/item_concated_fc/i_concated_fc/BiasAdd_grad/BiasAddGrad<^gradients/item_concated_fc/i_concated_fc/Relu_grad/ReluGrad
?
Ngradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/control_dependencyIdentity;gradients/item_concated_fc/i_concated_fc/Relu_grad/ReluGradG^gradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/item_concated_fc/i_concated_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????@
?
Pgradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/item_concated_fc/i_concated_fc/BiasAdd_grad/BiasAddGradG^gradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/item_concated_fc/i_concated_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
4gradients/user_fc/u_concated_fc/Tensordot_grad/ShapeShape&user_fc/u_concated_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
6gradients/user_fc/u_concated_fc/Tensordot_grad/ReshapeReshapeEgradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/control_dependency4gradients/user_fc/u_concated_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????@
?
=gradients/item_concated_fc/i_concated_fc/Tensordot_grad/ShapeShape/item_concated_fc/i_concated_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
?gradients/item_concated_fc/i_concated_fc/Tensordot_grad/ReshapeReshapeNgradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/control_dependency=gradients/item_concated_fc/i_concated_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????@
?
<gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMulMatMul6gradients/user_fc/u_concated_fc/Tensordot_grad/Reshape)user_fc/u_concated_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????<
?
>gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul_1MatMul'user_fc/u_concated_fc/Tensordot/Reshape6gradients/user_fc/u_concated_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????@
?
Fgradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp=^gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul?^gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul_1
?
Ngradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity<gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMulG^gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????<
?
Pgradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity>gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul_1G^gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:<@
?
Egradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMulMatMul?gradients/item_concated_fc/i_concated_fc/Tensordot_grad/Reshape2item_concated_fc/i_concated_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
Ggradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMul_1MatMul0item_concated_fc/i_concated_fc/Tensordot/Reshape?gradients/item_concated_fc/i_concated_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????@
?
Ogradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/group_depsNoOpF^gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMulH^gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMul_1
?
Wgradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentityEgradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMulP^gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Ygradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/control_dependency_1IdentityGgradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMul_1P^gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:@
?
<gradients/user_fc/u_concated_fc/Tensordot/Reshape_grad/ShapeShape)user_fc/u_concated_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
>gradients/user_fc/u_concated_fc/Tensordot/Reshape_grad/ReshapeReshapeNgradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/control_dependency<gradients/user_fc/u_concated_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????<
?
>gradients/user_fc/u_concated_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
?
@gradients/user_fc/u_concated_fc/Tensordot/Reshape_1_grad/ReshapeReshapePgradients/user_fc/u_concated_fc/Tensordot/MatMul_grad/tuple/control_dependency_1>gradients/user_fc/u_concated_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:<@
?
Egradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_grad/ShapeShape2item_concated_fc/i_concated_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
Ggradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_grad/ReshapeReshapeWgradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/control_dependencyEgradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
Ggradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
?
Igradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_1_grad/ReshapeReshapeYgradients/item_concated_fc/i_concated_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Ggradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:@
?
Jgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation&user_fc/u_concated_fc/Tensordot/concat*
T0*
_output_shapes
:
?
Bgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose	Transpose>gradients/user_fc/u_concated_fc/Tensordot/Reshape_grad/ReshapeJgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????<
?
Lgradients/user_fc/u_concated_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation0user_fc/u_concated_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
Dgradients/user_fc/u_concated_fc/Tensordot/transpose_1_grad/transpose	Transpose@gradients/user_fc/u_concated_fc/Tensordot/Reshape_1_grad/ReshapeLgradients/user_fc/u_concated_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:<@
?
Sgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation/item_concated_fc/i_concated_fc/Tensordot/concat*
T0*
_output_shapes
:
?
Kgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_grad/transpose	TransposeGgradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_grad/ReshapeSgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????
?
Ugradients/item_concated_fc/i_concated_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation9item_concated_fc/i_concated_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
Mgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_1_grad/transpose	TransposeIgradients/item_concated_fc/i_concated_fc/Tensordot/Reshape_1_grad/ReshapeUgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:@
h
&gradients/user_fc/u_concated_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients/user_fc/u_concated_grad/modFloorModuser_fc/u_concated/axis&gradients/user_fc/u_concated_grad/Rank*
T0*
_output_shapes
: 
}
'gradients/user_fc/u_concated_grad/ShapeShapeuser_fc/u_type_fc/Relu*
T0*
out_type0*
_output_shapes
:
?
(gradients/user_fc/u_concated_grad/ShapeNShapeNuser_fc/u_type_fc/Reluuser_fc/u_age_fc/Reluuser_fc/u_sex_fc/Reluuser_fc/u_org_fc/Reluuser_fc/u_seat_fc/Reluuser_fc/u_pos_id/Relu*
T0*
out_type0*
N*8
_output_shapes&
$::::::
?
.gradients/user_fc/u_concated_grad/ConcatOffsetConcatOffset%gradients/user_fc/u_concated_grad/mod(gradients/user_fc/u_concated_grad/ShapeN*gradients/user_fc/u_concated_grad/ShapeN:1*gradients/user_fc/u_concated_grad/ShapeN:2*gradients/user_fc/u_concated_grad/ShapeN:3*gradients/user_fc/u_concated_grad/ShapeN:4*gradients/user_fc/u_concated_grad/ShapeN:5*
N*8
_output_shapes&
$::::::
?
'gradients/user_fc/u_concated_grad/SliceSliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose.gradients/user_fc/u_concated_grad/ConcatOffset(gradients/user_fc/u_concated_grad/ShapeN*
T0*
Index0*+
_output_shapes
:?????????

?
)gradients/user_fc/u_concated_grad/Slice_1SliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose0gradients/user_fc/u_concated_grad/ConcatOffset:1*gradients/user_fc/u_concated_grad/ShapeN:1*
T0*
Index0*+
_output_shapes
:?????????

?
)gradients/user_fc/u_concated_grad/Slice_2SliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose0gradients/user_fc/u_concated_grad/ConcatOffset:2*gradients/user_fc/u_concated_grad/ShapeN:2*
T0*
Index0*+
_output_shapes
:?????????

?
)gradients/user_fc/u_concated_grad/Slice_3SliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose0gradients/user_fc/u_concated_grad/ConcatOffset:3*gradients/user_fc/u_concated_grad/ShapeN:3*
T0*
Index0*+
_output_shapes
:?????????

?
)gradients/user_fc/u_concated_grad/Slice_4SliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose0gradients/user_fc/u_concated_grad/ConcatOffset:4*gradients/user_fc/u_concated_grad/ShapeN:4*
T0*
Index0*+
_output_shapes
:?????????

?
)gradients/user_fc/u_concated_grad/Slice_5SliceBgradients/user_fc/u_concated_fc/Tensordot/transpose_grad/transpose0gradients/user_fc/u_concated_grad/ConcatOffset:5*gradients/user_fc/u_concated_grad/ShapeN:5*
T0*
Index0*+
_output_shapes
:?????????

?
2gradients/user_fc/u_concated_grad/tuple/group_depsNoOp(^gradients/user_fc/u_concated_grad/Slice*^gradients/user_fc/u_concated_grad/Slice_1*^gradients/user_fc/u_concated_grad/Slice_2*^gradients/user_fc/u_concated_grad/Slice_3*^gradients/user_fc/u_concated_grad/Slice_4*^gradients/user_fc/u_concated_grad/Slice_5
?
:gradients/user_fc/u_concated_grad/tuple/control_dependencyIdentity'gradients/user_fc/u_concated_grad/Slice3^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/user_fc/u_concated_grad/Slice*+
_output_shapes
:?????????

?
<gradients/user_fc/u_concated_grad/tuple/control_dependency_1Identity)gradients/user_fc/u_concated_grad/Slice_13^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/user_fc/u_concated_grad/Slice_1*+
_output_shapes
:?????????

?
<gradients/user_fc/u_concated_grad/tuple/control_dependency_2Identity)gradients/user_fc/u_concated_grad/Slice_23^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/user_fc/u_concated_grad/Slice_2*+
_output_shapes
:?????????

?
<gradients/user_fc/u_concated_grad/tuple/control_dependency_3Identity)gradients/user_fc/u_concated_grad/Slice_33^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/user_fc/u_concated_grad/Slice_3*+
_output_shapes
:?????????

?
<gradients/user_fc/u_concated_grad/tuple/control_dependency_4Identity)gradients/user_fc/u_concated_grad/Slice_43^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/user_fc/u_concated_grad/Slice_4*+
_output_shapes
:?????????

?
<gradients/user_fc/u_concated_grad/tuple/control_dependency_5Identity)gradients/user_fc/u_concated_grad/Slice_53^gradients/user_fc/u_concated_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/user_fc/u_concated_grad/Slice_5*+
_output_shapes
:?????????

?
5gradients/item_class_fc/i_class_fc/Relu_grad/ReluGradReluGradKgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_grad/transposeitem_class_fc/i_class_fc/Relu*
T0*+
_output_shapes
:?????????
?
.gradients/user_fc/u_type_fc/Relu_grad/ReluGradReluGrad:gradients/user_fc/u_concated_grad/tuple/control_dependencyuser_fc/u_type_fc/Relu*
T0*+
_output_shapes
:?????????

?
-gradients/user_fc/u_age_fc/Relu_grad/ReluGradReluGrad<gradients/user_fc/u_concated_grad/tuple/control_dependency_1user_fc/u_age_fc/Relu*
T0*+
_output_shapes
:?????????

?
-gradients/user_fc/u_sex_fc/Relu_grad/ReluGradReluGrad<gradients/user_fc/u_concated_grad/tuple/control_dependency_2user_fc/u_sex_fc/Relu*
T0*+
_output_shapes
:?????????

?
-gradients/user_fc/u_org_fc/Relu_grad/ReluGradReluGrad<gradients/user_fc/u_concated_grad/tuple/control_dependency_3user_fc/u_org_fc/Relu*
T0*+
_output_shapes
:?????????

?
.gradients/user_fc/u_seat_fc/Relu_grad/ReluGradReluGrad<gradients/user_fc/u_concated_grad/tuple/control_dependency_4user_fc/u_seat_fc/Relu*
T0*+
_output_shapes
:?????????

?
-gradients/user_fc/u_pos_id/Relu_grad/ReluGradReluGrad<gradients/user_fc/u_concated_grad/tuple/control_dependency_5user_fc/u_pos_id/Relu*
T0*+
_output_shapes
:?????????

?
;gradients/item_class_fc/i_class_fc/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/item_class_fc/i_class_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
?
@gradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/group_depsNoOp<^gradients/item_class_fc/i_class_fc/BiasAdd_grad/BiasAddGrad6^gradients/item_class_fc/i_class_fc/Relu_grad/ReluGrad
?
Hgradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/item_class_fc/i_class_fc/Relu_grad/ReluGradA^gradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/item_class_fc/i_class_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????
?
Jgradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/item_class_fc/i_class_fc/BiasAdd_grad/BiasAddGradA^gradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/item_class_fc/i_class_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
4gradients/user_fc/u_type_fc/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/user_fc/u_type_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
9gradients/user_fc/u_type_fc/BiasAdd_grad/tuple/group_depsNoOp5^gradients/user_fc/u_type_fc/BiasAdd_grad/BiasAddGrad/^gradients/user_fc/u_type_fc/Relu_grad/ReluGrad
?
Agradients/user_fc/u_type_fc/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/user_fc/u_type_fc/Relu_grad/ReluGrad:^gradients/user_fc/u_type_fc/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/user_fc/u_type_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Cgradients/user_fc/u_type_fc/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/user_fc/u_type_fc/BiasAdd_grad/BiasAddGrad:^gradients/user_fc/u_type_fc/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/user_fc/u_type_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
3gradients/user_fc/u_age_fc/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/user_fc/u_age_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
8gradients/user_fc/u_age_fc/BiasAdd_grad/tuple/group_depsNoOp4^gradients/user_fc/u_age_fc/BiasAdd_grad/BiasAddGrad.^gradients/user_fc/u_age_fc/Relu_grad/ReluGrad
?
@gradients/user_fc/u_age_fc/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/user_fc/u_age_fc/Relu_grad/ReluGrad9^gradients/user_fc/u_age_fc/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/user_fc/u_age_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Bgradients/user_fc/u_age_fc/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/user_fc/u_age_fc/BiasAdd_grad/BiasAddGrad9^gradients/user_fc/u_age_fc/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/user_fc/u_age_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
3gradients/user_fc/u_sex_fc/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/user_fc/u_sex_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
8gradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/group_depsNoOp4^gradients/user_fc/u_sex_fc/BiasAdd_grad/BiasAddGrad.^gradients/user_fc/u_sex_fc/Relu_grad/ReluGrad
?
@gradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/user_fc/u_sex_fc/Relu_grad/ReluGrad9^gradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/user_fc/u_sex_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Bgradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/user_fc/u_sex_fc/BiasAdd_grad/BiasAddGrad9^gradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/user_fc/u_sex_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
3gradients/user_fc/u_org_fc/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/user_fc/u_org_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
8gradients/user_fc/u_org_fc/BiasAdd_grad/tuple/group_depsNoOp4^gradients/user_fc/u_org_fc/BiasAdd_grad/BiasAddGrad.^gradients/user_fc/u_org_fc/Relu_grad/ReluGrad
?
@gradients/user_fc/u_org_fc/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/user_fc/u_org_fc/Relu_grad/ReluGrad9^gradients/user_fc/u_org_fc/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/user_fc/u_org_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Bgradients/user_fc/u_org_fc/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/user_fc/u_org_fc/BiasAdd_grad/BiasAddGrad9^gradients/user_fc/u_org_fc/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/user_fc/u_org_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
4gradients/user_fc/u_seat_fc/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/user_fc/u_seat_fc/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
9gradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/group_depsNoOp5^gradients/user_fc/u_seat_fc/BiasAdd_grad/BiasAddGrad/^gradients/user_fc/u_seat_fc/Relu_grad/ReluGrad
?
Agradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/user_fc/u_seat_fc/Relu_grad/ReluGrad:^gradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/user_fc/u_seat_fc/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Cgradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/user_fc/u_seat_fc/BiasAdd_grad/BiasAddGrad:^gradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/user_fc/u_seat_fc/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
3gradients/user_fc/u_pos_id/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/user_fc/u_pos_id/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:

?
8gradients/user_fc/u_pos_id/BiasAdd_grad/tuple/group_depsNoOp4^gradients/user_fc/u_pos_id/BiasAdd_grad/BiasAddGrad.^gradients/user_fc/u_pos_id/Relu_grad/ReluGrad
?
@gradients/user_fc/u_pos_id/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/user_fc/u_pos_id/Relu_grad/ReluGrad9^gradients/user_fc/u_pos_id/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/user_fc/u_pos_id/Relu_grad/ReluGrad*+
_output_shapes
:?????????

?
Bgradients/user_fc/u_pos_id/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/user_fc/u_pos_id/BiasAdd_grad/BiasAddGrad9^gradients/user_fc/u_pos_id/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/user_fc/u_pos_id/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

?
7gradients/item_class_fc/i_class_fc/Tensordot_grad/ShapeShape)item_class_fc/i_class_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
9gradients/item_class_fc/i_class_fc/Tensordot_grad/ReshapeReshapeHgradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/control_dependency7gradients/item_class_fc/i_class_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
0gradients/user_fc/u_type_fc/Tensordot_grad/ShapeShape"user_fc/u_type_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
2gradients/user_fc/u_type_fc/Tensordot_grad/ReshapeReshapeAgradients/user_fc/u_type_fc/BiasAdd_grad/tuple/control_dependency0gradients/user_fc/u_type_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
/gradients/user_fc/u_age_fc/Tensordot_grad/ShapeShape!user_fc/u_age_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
1gradients/user_fc/u_age_fc/Tensordot_grad/ReshapeReshape@gradients/user_fc/u_age_fc/BiasAdd_grad/tuple/control_dependency/gradients/user_fc/u_age_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
/gradients/user_fc/u_sex_fc/Tensordot_grad/ShapeShape!user_fc/u_sex_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
1gradients/user_fc/u_sex_fc/Tensordot_grad/ReshapeReshape@gradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/control_dependency/gradients/user_fc/u_sex_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
/gradients/user_fc/u_org_fc/Tensordot_grad/ShapeShape!user_fc/u_org_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
1gradients/user_fc/u_org_fc/Tensordot_grad/ReshapeReshape@gradients/user_fc/u_org_fc/BiasAdd_grad/tuple/control_dependency/gradients/user_fc/u_org_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
0gradients/user_fc/u_seat_fc/Tensordot_grad/ShapeShape"user_fc/u_seat_fc/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
2gradients/user_fc/u_seat_fc/Tensordot_grad/ReshapeReshapeAgradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/control_dependency0gradients/user_fc/u_seat_fc/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
/gradients/user_fc/u_pos_id/Tensordot_grad/ShapeShape!user_fc/u_pos_id/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
?
1gradients/user_fc/u_pos_id/Tensordot_grad/ReshapeReshape@gradients/user_fc/u_pos_id/BiasAdd_grad/tuple/control_dependency/gradients/user_fc/u_pos_id/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

?
?gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMulMatMul9gradients/item_class_fc/i_class_fc/Tensordot_grad/Reshape,item_class_fc/i_class_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
Agradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMul_1MatMul*item_class_fc/i_class_fc/Tensordot/Reshape9gradients/item_class_fc/i_class_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????
?
Igradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp@^gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMulB^gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMul_1
?
Qgradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity?gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMulJ^gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
Sgradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/control_dependency_1IdentityAgradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMul_1J^gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:
?
8gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMulMatMul2gradients/user_fc/u_type_fc/Tensordot_grad/Reshape%user_fc/u_type_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
:gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul_1MatMul#user_fc/u_type_fc/Tensordot/Reshape2gradients/user_fc/u_type_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Bgradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp9^gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul;^gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul_1
?
Jgradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity8gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMulC^gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Lgradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity:gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul_1C^gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/user_fc/u_type_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
7gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMulMatMul1gradients/user_fc/u_age_fc/Tensordot_grad/Reshape$user_fc/u_age_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
9gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul_1MatMul"user_fc/u_age_fc/Tensordot/Reshape1gradients/user_fc/u_age_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Agradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp8^gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul:^gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul_1
?
Igradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity7gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMulB^gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Kgradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity9gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul_1B^gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/user_fc/u_age_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
7gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMulMatMul1gradients/user_fc/u_sex_fc/Tensordot_grad/Reshape$user_fc/u_sex_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
9gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul_1MatMul"user_fc/u_sex_fc/Tensordot/Reshape1gradients/user_fc/u_sex_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Agradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp8^gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul:^gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul_1
?
Igradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity7gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMulB^gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Kgradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity9gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul_1B^gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
7gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMulMatMul1gradients/user_fc/u_org_fc/Tensordot_grad/Reshape$user_fc/u_org_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
9gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul_1MatMul"user_fc/u_org_fc/Tensordot/Reshape1gradients/user_fc/u_org_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Agradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp8^gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul:^gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul_1
?
Igradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity7gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMulB^gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Kgradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity9gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul_1B^gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/user_fc/u_org_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
8gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMulMatMul2gradients/user_fc/u_seat_fc/Tensordot_grad/Reshape%user_fc/u_seat_fc/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
:gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul_1MatMul#user_fc/u_seat_fc/Tensordot/Reshape2gradients/user_fc/u_seat_fc/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Bgradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/group_depsNoOp9^gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul;^gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul_1
?
Jgradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/control_dependencyIdentity8gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMulC^gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Lgradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Identity:gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul_1C^gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
7gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMulMatMul1gradients/user_fc/u_pos_id/Tensordot_grad/Reshape$user_fc/u_pos_id/Tensordot/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????

?
9gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul_1MatMul"user_fc/u_pos_id/Tensordot/Reshape1gradients/user_fc/u_pos_id/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:?????????

?
Agradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/group_depsNoOp8^gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul:^gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul_1
?
Igradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/control_dependencyIdentity7gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMulB^gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:?????????

?
Kgradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/control_dependency_1Identity9gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul_1B^gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/user_fc/u_pos_id/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:


?
?gradients/item_class_fc/i_class_fc/Tensordot/Reshape_grad/ShapeShape,item_class_fc/i_class_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
Agradients/item_class_fc/i_class_fc/Tensordot/Reshape_grad/ReshapeReshapeQgradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/control_dependency?gradients/item_class_fc/i_class_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
Agradients/item_class_fc/i_class_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
Cgradients/item_class_fc/i_class_fc/Tensordot/Reshape_1_grad/ReshapeReshapeSgradients/item_class_fc/i_class_fc/Tensordot/MatMul_grad/tuple/control_dependency_1Agradients/item_class_fc/i_class_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:
?
8gradients/user_fc/u_type_fc/Tensordot/Reshape_grad/ShapeShape%user_fc/u_type_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
:gradients/user_fc/u_type_fc/Tensordot/Reshape_grad/ReshapeReshapeJgradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/control_dependency8gradients/user_fc/u_type_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
:gradients/user_fc/u_type_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
<gradients/user_fc/u_type_fc/Tensordot/Reshape_1_grad/ReshapeReshapeLgradients/user_fc/u_type_fc/Tensordot/MatMul_grad/tuple/control_dependency_1:gradients/user_fc/u_type_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
7gradients/user_fc/u_age_fc/Tensordot/Reshape_grad/ShapeShape$user_fc/u_age_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
9gradients/user_fc/u_age_fc/Tensordot/Reshape_grad/ReshapeReshapeIgradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/control_dependency7gradients/user_fc/u_age_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
9gradients/user_fc/u_age_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
;gradients/user_fc/u_age_fc/Tensordot/Reshape_1_grad/ReshapeReshapeKgradients/user_fc/u_age_fc/Tensordot/MatMul_grad/tuple/control_dependency_19gradients/user_fc/u_age_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
7gradients/user_fc/u_sex_fc/Tensordot/Reshape_grad/ShapeShape$user_fc/u_sex_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
9gradients/user_fc/u_sex_fc/Tensordot/Reshape_grad/ReshapeReshapeIgradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/control_dependency7gradients/user_fc/u_sex_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
9gradients/user_fc/u_sex_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
;gradients/user_fc/u_sex_fc/Tensordot/Reshape_1_grad/ReshapeReshapeKgradients/user_fc/u_sex_fc/Tensordot/MatMul_grad/tuple/control_dependency_19gradients/user_fc/u_sex_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
7gradients/user_fc/u_org_fc/Tensordot/Reshape_grad/ShapeShape$user_fc/u_org_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
9gradients/user_fc/u_org_fc/Tensordot/Reshape_grad/ReshapeReshapeIgradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/control_dependency7gradients/user_fc/u_org_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
9gradients/user_fc/u_org_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
;gradients/user_fc/u_org_fc/Tensordot/Reshape_1_grad/ReshapeReshapeKgradients/user_fc/u_org_fc/Tensordot/MatMul_grad/tuple/control_dependency_19gradients/user_fc/u_org_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
8gradients/user_fc/u_seat_fc/Tensordot/Reshape_grad/ShapeShape%user_fc/u_seat_fc/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
:gradients/user_fc/u_seat_fc/Tensordot/Reshape_grad/ReshapeReshapeJgradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/control_dependency8gradients/user_fc/u_seat_fc/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
:gradients/user_fc/u_seat_fc/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
<gradients/user_fc/u_seat_fc/Tensordot/Reshape_1_grad/ReshapeReshapeLgradients/user_fc/u_seat_fc/Tensordot/MatMul_grad/tuple/control_dependency_1:gradients/user_fc/u_seat_fc/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
7gradients/user_fc/u_pos_id/Tensordot/Reshape_grad/ShapeShape$user_fc/u_pos_id/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
?
9gradients/user_fc/u_pos_id/Tensordot/Reshape_grad/ReshapeReshapeIgradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/control_dependency7gradients/user_fc/u_pos_id/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????

?
9gradients/user_fc/u_pos_id/Tensordot/Reshape_1_grad/ShapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
?
;gradients/user_fc/u_pos_id/Tensordot/Reshape_1_grad/ReshapeReshapeKgradients/user_fc/u_pos_id/Tensordot/MatMul_grad/tuple/control_dependency_19gradients/user_fc/u_pos_id/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:


?
Mgradients/item_class_fc/i_class_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation)item_class_fc/i_class_fc/Tensordot/concat*
T0*
_output_shapes
:
?
Egradients/item_class_fc/i_class_fc/Tensordot/transpose_grad/transpose	TransposeAgradients/item_class_fc/i_class_fc/Tensordot/Reshape_grad/ReshapeMgradients/item_class_fc/i_class_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????
?
Ogradients/item_class_fc/i_class_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation3item_class_fc/i_class_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
Ggradients/item_class_fc/i_class_fc/Tensordot/transpose_1_grad/transpose	TransposeCgradients/item_class_fc/i_class_fc/Tensordot/Reshape_1_grad/ReshapeOgradients/item_class_fc/i_class_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:
?
Fgradients/user_fc/u_type_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation"user_fc/u_type_fc/Tensordot/concat*
T0*
_output_shapes
:
?
>gradients/user_fc/u_type_fc/Tensordot/transpose_grad/transpose	Transpose:gradients/user_fc/u_type_fc/Tensordot/Reshape_grad/ReshapeFgradients/user_fc/u_type_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Hgradients/user_fc/u_type_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation,user_fc/u_type_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
@gradients/user_fc/u_type_fc/Tensordot/transpose_1_grad/transpose	Transpose<gradients/user_fc/u_type_fc/Tensordot/Reshape_1_grad/ReshapeHgradients/user_fc/u_type_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
Egradients/user_fc/u_age_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation!user_fc/u_age_fc/Tensordot/concat*
T0*
_output_shapes
:
?
=gradients/user_fc/u_age_fc/Tensordot/transpose_grad/transpose	Transpose9gradients/user_fc/u_age_fc/Tensordot/Reshape_grad/ReshapeEgradients/user_fc/u_age_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Ggradients/user_fc/u_age_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation+user_fc/u_age_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
?gradients/user_fc/u_age_fc/Tensordot/transpose_1_grad/transpose	Transpose;gradients/user_fc/u_age_fc/Tensordot/Reshape_1_grad/ReshapeGgradients/user_fc/u_age_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
Egradients/user_fc/u_sex_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation!user_fc/u_sex_fc/Tensordot/concat*
T0*
_output_shapes
:
?
=gradients/user_fc/u_sex_fc/Tensordot/transpose_grad/transpose	Transpose9gradients/user_fc/u_sex_fc/Tensordot/Reshape_grad/ReshapeEgradients/user_fc/u_sex_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Ggradients/user_fc/u_sex_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation+user_fc/u_sex_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
?gradients/user_fc/u_sex_fc/Tensordot/transpose_1_grad/transpose	Transpose;gradients/user_fc/u_sex_fc/Tensordot/Reshape_1_grad/ReshapeGgradients/user_fc/u_sex_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
Egradients/user_fc/u_org_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation!user_fc/u_org_fc/Tensordot/concat*
T0*
_output_shapes
:
?
=gradients/user_fc/u_org_fc/Tensordot/transpose_grad/transpose	Transpose9gradients/user_fc/u_org_fc/Tensordot/Reshape_grad/ReshapeEgradients/user_fc/u_org_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Ggradients/user_fc/u_org_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation+user_fc/u_org_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
?gradients/user_fc/u_org_fc/Tensordot/transpose_1_grad/transpose	Transpose;gradients/user_fc/u_org_fc/Tensordot/Reshape_1_grad/ReshapeGgradients/user_fc/u_org_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
Fgradients/user_fc/u_seat_fc/Tensordot/transpose_grad/InvertPermutationInvertPermutation"user_fc/u_seat_fc/Tensordot/concat*
T0*
_output_shapes
:
?
>gradients/user_fc/u_seat_fc/Tensordot/transpose_grad/transpose	Transpose:gradients/user_fc/u_seat_fc/Tensordot/Reshape_grad/ReshapeFgradients/user_fc/u_seat_fc/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Hgradients/user_fc/u_seat_fc/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation,user_fc/u_seat_fc/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
@gradients/user_fc/u_seat_fc/Tensordot/transpose_1_grad/transpose	Transpose<gradients/user_fc/u_seat_fc/Tensordot/Reshape_1_grad/ReshapeHgradients/user_fc/u_seat_fc/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
Egradients/user_fc/u_pos_id/Tensordot/transpose_grad/InvertPermutationInvertPermutation!user_fc/u_pos_id/Tensordot/concat*
T0*
_output_shapes
:
?
=gradients/user_fc/u_pos_id/Tensordot/transpose_grad/transpose	Transpose9gradients/user_fc/u_pos_id/Tensordot/Reshape_grad/ReshapeEgradients/user_fc/u_pos_id/Tensordot/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:?????????

?
Ggradients/user_fc/u_pos_id/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation+user_fc/u_pos_id/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
?
?gradients/user_fc/u_pos_id/Tensordot/transpose_1_grad/transpose	Transpose;gradients/user_fc/u_pos_id/Tensordot/Reshape_1_grad/ReshapeGgradients/user_fc/u_pos_id/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:


?
;gradients/item_class_embedding/i_class_emb_layer_grad/ShapeConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*%
valueB	"              *
dtype0	*
_output_shapes
:
?
:gradients/item_class_embedding/i_class_emb_layer_grad/CastCast;gradients/item_class_embedding/i_class_emb_layer_grad/Shape*

SrcT0	*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
?
:gradients/item_class_embedding/i_class_emb_layer_grad/SizeSizei_class_label*
T0*
out_type0*
_output_shapes
: 
?
Dgradients/item_class_embedding/i_class_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
@gradients/item_class_embedding/i_class_emb_layer_grad/ExpandDims
ExpandDims:gradients/item_class_embedding/i_class_emb_layer_grad/SizeDgradients/item_class_embedding/i_class_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Igradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Kgradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Kgradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients/item_class_embedding/i_class_emb_layer_grad/strided_sliceStridedSlice:gradients/item_class_embedding/i_class_emb_layer_grad/CastIgradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stackKgradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stack_1Kgradients/item_class_embedding/i_class_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
?
Agradients/item_class_embedding/i_class_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
<gradients/item_class_embedding/i_class_emb_layer_grad/concatConcatV2@gradients/item_class_embedding/i_class_emb_layer_grad/ExpandDimsCgradients/item_class_embedding/i_class_emb_layer_grad/strided_sliceAgradients/item_class_embedding/i_class_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
=gradients/item_class_embedding/i_class_emb_layer_grad/ReshapeReshapeEgradients/item_class_fc/i_class_fc/Tensordot/transpose_grad/transpose<gradients/item_class_embedding/i_class_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
?gradients/item_class_embedding/i_class_emb_layer_grad/Reshape_1Reshapei_class_label@gradients/item_class_embedding/i_class_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
4gradients/user_embedding/u_type_emb_layer_grad/ShapeConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
3gradients/user_embedding/u_type_emb_layer_grad/CastCast4gradients/user_embedding/u_type_emb_layer_grad/Shape*

SrcT0	*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
t
3gradients/user_embedding/u_type_emb_layer_grad/SizeSizeu_type*
T0*
out_type0*
_output_shapes
: 

=gradients/user_embedding/u_type_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
9gradients/user_embedding/u_type_emb_layer_grad/ExpandDims
ExpandDims3gradients/user_embedding/u_type_emb_layer_grad/Size=gradients/user_embedding/u_type_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Bgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Dgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Dgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
<gradients/user_embedding/u_type_emb_layer_grad/strided_sliceStridedSlice3gradients/user_embedding/u_type_emb_layer_grad/CastBgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stackDgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stack_1Dgradients/user_embedding/u_type_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
|
:gradients/user_embedding/u_type_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
5gradients/user_embedding/u_type_emb_layer_grad/concatConcatV29gradients/user_embedding/u_type_emb_layer_grad/ExpandDims<gradients/user_embedding/u_type_emb_layer_grad/strided_slice:gradients/user_embedding/u_type_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
6gradients/user_embedding/u_type_emb_layer_grad/ReshapeReshape>gradients/user_fc/u_type_fc/Tensordot/transpose_grad/transpose5gradients/user_embedding/u_type_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
8gradients/user_embedding/u_type_emb_layer_grad/Reshape_1Reshapeu_type9gradients/user_embedding/u_type_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
3gradients/user_embedding/u_age_emb_layer_grad/ShapeConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
2gradients/user_embedding/u_age_emb_layer_grad/CastCast3gradients/user_embedding/u_age_emb_layer_grad/Shape*

SrcT0	*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
Truncate( *

DstT0*
_output_shapes
:
r
2gradients/user_embedding/u_age_emb_layer_grad/SizeSizeu_age*
T0*
out_type0*
_output_shapes
: 
~
<gradients/user_embedding/u_age_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
8gradients/user_embedding/u_age_emb_layer_grad/ExpandDims
ExpandDims2gradients/user_embedding/u_age_emb_layer_grad/Size<gradients/user_embedding/u_age_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Agradients/user_embedding/u_age_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_age_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_age_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
;gradients/user_embedding/u_age_emb_layer_grad/strided_sliceStridedSlice2gradients/user_embedding/u_age_emb_layer_grad/CastAgradients/user_embedding/u_age_emb_layer_grad/strided_slice/stackCgradients/user_embedding/u_age_emb_layer_grad/strided_slice/stack_1Cgradients/user_embedding/u_age_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
{
9gradients/user_embedding/u_age_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
4gradients/user_embedding/u_age_emb_layer_grad/concatConcatV28gradients/user_embedding/u_age_emb_layer_grad/ExpandDims;gradients/user_embedding/u_age_emb_layer_grad/strided_slice9gradients/user_embedding/u_age_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
5gradients/user_embedding/u_age_emb_layer_grad/ReshapeReshape=gradients/user_fc/u_age_fc/Tensordot/transpose_grad/transpose4gradients/user_embedding/u_age_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
7gradients/user_embedding/u_age_emb_layer_grad/Reshape_1Reshapeu_age8gradients/user_embedding/u_age_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
3gradients/user_embedding/u_sex_emb_layer_grad/ShapeConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
2gradients/user_embedding/u_sex_emb_layer_grad/CastCast3gradients/user_embedding/u_sex_emb_layer_grad/Shape*

SrcT0	*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
r
2gradients/user_embedding/u_sex_emb_layer_grad/SizeSizeu_sex*
T0*
out_type0*
_output_shapes
: 
~
<gradients/user_embedding/u_sex_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
8gradients/user_embedding/u_sex_emb_layer_grad/ExpandDims
ExpandDims2gradients/user_embedding/u_sex_emb_layer_grad/Size<gradients/user_embedding/u_sex_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Agradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
;gradients/user_embedding/u_sex_emb_layer_grad/strided_sliceStridedSlice2gradients/user_embedding/u_sex_emb_layer_grad/CastAgradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stackCgradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stack_1Cgradients/user_embedding/u_sex_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
{
9gradients/user_embedding/u_sex_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
4gradients/user_embedding/u_sex_emb_layer_grad/concatConcatV28gradients/user_embedding/u_sex_emb_layer_grad/ExpandDims;gradients/user_embedding/u_sex_emb_layer_grad/strided_slice9gradients/user_embedding/u_sex_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
5gradients/user_embedding/u_sex_emb_layer_grad/ReshapeReshape=gradients/user_fc/u_sex_fc/Tensordot/transpose_grad/transpose4gradients/user_embedding/u_sex_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
7gradients/user_embedding/u_sex_emb_layer_grad/Reshape_1Reshapeu_sex8gradients/user_embedding/u_sex_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
3gradients/user_embedding/u_org_emb_layer_grad/ShapeConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
2gradients/user_embedding/u_org_emb_layer_grad/CastCast3gradients/user_embedding/u_org_emb_layer_grad/Shape*

SrcT0	*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
u
2gradients/user_embedding/u_org_emb_layer_grad/SizeSizeu_org_id*
T0*
out_type0*
_output_shapes
: 
~
<gradients/user_embedding/u_org_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
8gradients/user_embedding/u_org_emb_layer_grad/ExpandDims
ExpandDims2gradients/user_embedding/u_org_emb_layer_grad/Size<gradients/user_embedding/u_org_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Agradients/user_embedding/u_org_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_org_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_org_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
;gradients/user_embedding/u_org_emb_layer_grad/strided_sliceStridedSlice2gradients/user_embedding/u_org_emb_layer_grad/CastAgradients/user_embedding/u_org_emb_layer_grad/strided_slice/stackCgradients/user_embedding/u_org_emb_layer_grad/strided_slice/stack_1Cgradients/user_embedding/u_org_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
{
9gradients/user_embedding/u_org_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
4gradients/user_embedding/u_org_emb_layer_grad/concatConcatV28gradients/user_embedding/u_org_emb_layer_grad/ExpandDims;gradients/user_embedding/u_org_emb_layer_grad/strided_slice9gradients/user_embedding/u_org_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
5gradients/user_embedding/u_org_emb_layer_grad/ReshapeReshape=gradients/user_fc/u_org_fc/Tensordot/transpose_grad/transpose4gradients/user_embedding/u_org_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
7gradients/user_embedding/u_org_emb_layer_grad/Reshape_1Reshapeu_org_id8gradients/user_embedding/u_org_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
4gradients/user_embedding/u_seat_emb_layer_grad/ShapeConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
3gradients/user_embedding/u_seat_emb_layer_grad/CastCast4gradients/user_embedding/u_seat_emb_layer_grad/Shape*

SrcT0	*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
w
3gradients/user_embedding/u_seat_emb_layer_grad/SizeSize	u_seat_id*
T0*
out_type0*
_output_shapes
: 

=gradients/user_embedding/u_seat_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
9gradients/user_embedding/u_seat_emb_layer_grad/ExpandDims
ExpandDims3gradients/user_embedding/u_seat_emb_layer_grad/Size=gradients/user_embedding/u_seat_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Bgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Dgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Dgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
<gradients/user_embedding/u_seat_emb_layer_grad/strided_sliceStridedSlice3gradients/user_embedding/u_seat_emb_layer_grad/CastBgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stackDgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stack_1Dgradients/user_embedding/u_seat_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
|
:gradients/user_embedding/u_seat_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
5gradients/user_embedding/u_seat_emb_layer_grad/concatConcatV29gradients/user_embedding/u_seat_emb_layer_grad/ExpandDims<gradients/user_embedding/u_seat_emb_layer_grad/strided_slice:gradients/user_embedding/u_seat_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
6gradients/user_embedding/u_seat_emb_layer_grad/ReshapeReshape>gradients/user_fc/u_seat_fc/Tensordot/transpose_grad/transpose5gradients/user_embedding/u_seat_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
8gradients/user_embedding/u_seat_emb_layer_grad/Reshape_1Reshape	u_seat_id9gradients/user_embedding/u_seat_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
?
3gradients/user_embedding/u_pos_emb_layer_grad/ShapeConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*%
valueB	"       
       *
dtype0	*
_output_shapes
:
?
2gradients/user_embedding/u_pos_emb_layer_grad/CastCast3gradients/user_embedding/u_pos_emb_layer_grad/Shape*

SrcT0	*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
Truncate( *

DstT0*
_output_shapes
:
u
2gradients/user_embedding/u_pos_emb_layer_grad/SizeSizeu_pos_id*
T0*
out_type0*
_output_shapes
: 
~
<gradients/user_embedding/u_pos_emb_layer_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
8gradients/user_embedding/u_pos_emb_layer_grad/ExpandDims
ExpandDims2gradients/user_embedding/u_pos_emb_layer_grad/Size<gradients/user_embedding/u_pos_emb_layer_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Agradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Cgradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
;gradients/user_embedding/u_pos_emb_layer_grad/strided_sliceStridedSlice2gradients/user_embedding/u_pos_emb_layer_grad/CastAgradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stackCgradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stack_1Cgradients/user_embedding/u_pos_emb_layer_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
{
9gradients/user_embedding/u_pos_emb_layer_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
4gradients/user_embedding/u_pos_emb_layer_grad/concatConcatV28gradients/user_embedding/u_pos_emb_layer_grad/ExpandDims;gradients/user_embedding/u_pos_emb_layer_grad/strided_slice9gradients/user_embedding/u_pos_emb_layer_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
5gradients/user_embedding/u_pos_emb_layer_grad/ReshapeReshape=gradients/user_fc/u_pos_id/Tensordot/transpose_grad/transpose4gradients/user_embedding/u_pos_emb_layer_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????

?
7gradients/user_embedding/u_pos_emb_layer_grad/Reshape_1Reshapeu_pos_id8gradients/user_embedding/u_pos_emb_layer_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
}
beta1_power/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
i
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
}
beta2_power/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *w??*
dtype0*
_output_shapes
: 
?
beta2_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
: 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
i
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
7user_embedding/u_type_emb_matrix/Adam/Initializer/zerosConst*
valueB
*    *3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
dtype0*
_output_shapes

:

?
%user_embedding/u_type_emb_matrix/Adam
VariableV2*
shape
:
*
shared_name *3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
,user_embedding/u_type_emb_matrix/Adam/AssignAssign%user_embedding/u_type_emb_matrix/Adam7user_embedding/u_type_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
*user_embedding/u_type_emb_matrix/Adam/readIdentity%user_embedding/u_type_emb_matrix/Adam*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
9user_embedding/u_type_emb_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
dtype0*
_output_shapes

:

?
'user_embedding/u_type_emb_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
.user_embedding/u_type_emb_matrix/Adam_1/AssignAssign'user_embedding/u_type_emb_matrix/Adam_19user_embedding/u_type_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
,user_embedding/u_type_emb_matrix/Adam_1/readIdentity'user_embedding/u_type_emb_matrix/Adam_1*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
6user_embedding/u_age_emn_matrix/Adam/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
dtype0*
_output_shapes

:

?
$user_embedding/u_age_emn_matrix/Adam
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
dtype0*
	container *
_output_shapes

:

?
+user_embedding/u_age_emn_matrix/Adam/AssignAssign$user_embedding/u_age_emn_matrix/Adam6user_embedding/u_age_emn_matrix/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
)user_embedding/u_age_emn_matrix/Adam/readIdentity$user_embedding/u_age_emn_matrix/Adam*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
8user_embedding/u_age_emn_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
dtype0*
_output_shapes

:

?
&user_embedding/u_age_emn_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
dtype0*
	container *
_output_shapes

:

?
-user_embedding/u_age_emn_matrix/Adam_1/AssignAssign&user_embedding/u_age_emn_matrix/Adam_18user_embedding/u_age_emn_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
+user_embedding/u_age_emn_matrix/Adam_1/readIdentity&user_embedding/u_age_emn_matrix/Adam_1*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
6user_embedding/u_sex_emb_matrix/Adam/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
dtype0*
_output_shapes

:

?
$user_embedding/u_sex_emb_matrix/Adam
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
+user_embedding/u_sex_emb_matrix/Adam/AssignAssign$user_embedding/u_sex_emb_matrix/Adam6user_embedding/u_sex_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
)user_embedding/u_sex_emb_matrix/Adam/readIdentity$user_embedding/u_sex_emb_matrix/Adam*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
8user_embedding/u_sex_emb_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
dtype0*
_output_shapes

:

?
&user_embedding/u_sex_emb_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
-user_embedding/u_sex_emb_matrix/Adam_1/AssignAssign&user_embedding/u_sex_emb_matrix/Adam_18user_embedding/u_sex_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
+user_embedding/u_sex_emb_matrix/Adam_1/readIdentity&user_embedding/u_sex_emb_matrix/Adam_1*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
6user_embedding/u_org_emb_matrix/Adam/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
dtype0*
_output_shapes

:

?
$user_embedding/u_org_emb_matrix/Adam
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
+user_embedding/u_org_emb_matrix/Adam/AssignAssign$user_embedding/u_org_emb_matrix/Adam6user_embedding/u_org_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
)user_embedding/u_org_emb_matrix/Adam/readIdentity$user_embedding/u_org_emb_matrix/Adam*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
8user_embedding/u_org_emb_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
dtype0*
_output_shapes

:

?
&user_embedding/u_org_emb_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
-user_embedding/u_org_emb_matrix/Adam_1/AssignAssign&user_embedding/u_org_emb_matrix/Adam_18user_embedding/u_org_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
+user_embedding/u_org_emb_matrix/Adam_1/readIdentity&user_embedding/u_org_emb_matrix/Adam_1*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
7user_embedding/u_seat_emb_matrix/Adam/Initializer/zerosConst*
valueB
*    *3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
dtype0*
_output_shapes

:

?
%user_embedding/u_seat_emb_matrix/Adam
VariableV2*
shape
:
*
shared_name *3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
,user_embedding/u_seat_emb_matrix/Adam/AssignAssign%user_embedding/u_seat_emb_matrix/Adam7user_embedding/u_seat_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
*user_embedding/u_seat_emb_matrix/Adam/readIdentity%user_embedding/u_seat_emb_matrix/Adam*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
9user_embedding/u_seat_emb_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
dtype0*
_output_shapes

:

?
'user_embedding/u_seat_emb_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
.user_embedding/u_seat_emb_matrix/Adam_1/AssignAssign'user_embedding/u_seat_emb_matrix/Adam_19user_embedding/u_seat_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
,user_embedding/u_seat_emb_matrix/Adam_1/readIdentity'user_embedding/u_seat_emb_matrix/Adam_1*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
6user_embedding/u_pos_emb_matrix/Adam/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
dtype0*
_output_shapes

:

?
$user_embedding/u_pos_emb_matrix/Adam
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
+user_embedding/u_pos_emb_matrix/Adam/AssignAssign$user_embedding/u_pos_emb_matrix/Adam6user_embedding/u_pos_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
)user_embedding/u_pos_emb_matrix/Adam/readIdentity$user_embedding/u_pos_emb_matrix/Adam*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
8user_embedding/u_pos_emb_matrix/Adam_1/Initializer/zerosConst*
valueB
*    *2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
dtype0*
_output_shapes

:

?
&user_embedding/u_pos_emb_matrix/Adam_1
VariableV2*
shape
:
*
shared_name *2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
dtype0*
	container *
_output_shapes

:

?
-user_embedding/u_pos_emb_matrix/Adam_1/AssignAssign&user_embedding/u_pos_emb_matrix/Adam_18user_embedding/u_pos_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
+user_embedding/u_pos_emb_matrix/Adam_1/readIdentity&user_embedding/u_pos_emb_matrix/Adam_1*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
'u_type_fc/kernel/Adam/Initializer/zerosConst*
valueB

*    *#
_class
loc:@u_type_fc/kernel*
dtype0*
_output_shapes

:


?
u_type_fc/kernel/Adam
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_type_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_type_fc/kernel/Adam/AssignAssignu_type_fc/kernel/Adam'u_type_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_type_fc/kernel/Adam/readIdentityu_type_fc/kernel/Adam*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes

:


?
)u_type_fc/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *#
_class
loc:@u_type_fc/kernel*
dtype0*
_output_shapes

:


?
u_type_fc/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_type_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_type_fc/kernel/Adam_1/AssignAssignu_type_fc/kernel/Adam_1)u_type_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_type_fc/kernel/Adam_1/readIdentityu_type_fc/kernel/Adam_1*
T0*#
_class
loc:@u_type_fc/kernel*
_output_shapes

:


?
%u_type_fc/bias/Adam/Initializer/zerosConst*
valueB
*    *!
_class
loc:@u_type_fc/bias*
dtype0*
_output_shapes
:

?
u_type_fc/bias/Adam
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_type_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_type_fc/bias/Adam/AssignAssignu_type_fc/bias/Adam%u_type_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

?
u_type_fc/bias/Adam/readIdentityu_type_fc/bias/Adam*
T0*!
_class
loc:@u_type_fc/bias*
_output_shapes
:

?
'u_type_fc/bias/Adam_1/Initializer/zerosConst*
valueB
*    *!
_class
loc:@u_type_fc/bias*
dtype0*
_output_shapes
:

?
u_type_fc/bias/Adam_1
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_type_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_type_fc/bias/Adam_1/AssignAssignu_type_fc/bias/Adam_1'u_type_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

?
u_type_fc/bias/Adam_1/readIdentityu_type_fc/bias/Adam_1*
T0*!
_class
loc:@u_type_fc/bias*
_output_shapes
:

?
&u_age_fc/kernel/Adam/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_age_fc/kernel*
dtype0*
_output_shapes

:


?
u_age_fc/kernel/Adam
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_age_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_age_fc/kernel/Adam/AssignAssignu_age_fc/kernel/Adam&u_age_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_age_fc/kernel/Adam/readIdentityu_age_fc/kernel/Adam*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes

:


?
(u_age_fc/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_age_fc/kernel*
dtype0*
_output_shapes

:


?
u_age_fc/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_age_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_age_fc/kernel/Adam_1/AssignAssignu_age_fc/kernel/Adam_1(u_age_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_age_fc/kernel/Adam_1/readIdentityu_age_fc/kernel/Adam_1*
T0*"
_class
loc:@u_age_fc/kernel*
_output_shapes

:


?
$u_age_fc/bias/Adam/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_age_fc/bias*
dtype0*
_output_shapes
:

?
u_age_fc/bias/Adam
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_age_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_age_fc/bias/Adam/AssignAssignu_age_fc/bias/Adam$u_age_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

~
u_age_fc/bias/Adam/readIdentityu_age_fc/bias/Adam*
T0* 
_class
loc:@u_age_fc/bias*
_output_shapes
:

?
&u_age_fc/bias/Adam_1/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_age_fc/bias*
dtype0*
_output_shapes
:

?
u_age_fc/bias/Adam_1
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_age_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_age_fc/bias/Adam_1/AssignAssignu_age_fc/bias/Adam_1&u_age_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

?
u_age_fc/bias/Adam_1/readIdentityu_age_fc/bias/Adam_1*
T0* 
_class
loc:@u_age_fc/bias*
_output_shapes
:

?
&u_sex_fc/kernel/Adam/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_sex_fc/kernel*
dtype0*
_output_shapes

:


?
u_sex_fc/kernel/Adam
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_sex_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_sex_fc/kernel/Adam/AssignAssignu_sex_fc/kernel/Adam&u_sex_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_sex_fc/kernel/Adam/readIdentityu_sex_fc/kernel/Adam*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes

:


?
(u_sex_fc/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_sex_fc/kernel*
dtype0*
_output_shapes

:


?
u_sex_fc/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_sex_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_sex_fc/kernel/Adam_1/AssignAssignu_sex_fc/kernel/Adam_1(u_sex_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_sex_fc/kernel/Adam_1/readIdentityu_sex_fc/kernel/Adam_1*
T0*"
_class
loc:@u_sex_fc/kernel*
_output_shapes

:


?
$u_sex_fc/bias/Adam/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_sex_fc/bias*
dtype0*
_output_shapes
:

?
u_sex_fc/bias/Adam
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_sex_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_sex_fc/bias/Adam/AssignAssignu_sex_fc/bias/Adam$u_sex_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

~
u_sex_fc/bias/Adam/readIdentityu_sex_fc/bias/Adam*
T0* 
_class
loc:@u_sex_fc/bias*
_output_shapes
:

?
&u_sex_fc/bias/Adam_1/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_sex_fc/bias*
dtype0*
_output_shapes
:

?
u_sex_fc/bias/Adam_1
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_sex_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_sex_fc/bias/Adam_1/AssignAssignu_sex_fc/bias/Adam_1&u_sex_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

?
u_sex_fc/bias/Adam_1/readIdentityu_sex_fc/bias/Adam_1*
T0* 
_class
loc:@u_sex_fc/bias*
_output_shapes
:

?
&u_org_fc/kernel/Adam/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_org_fc/kernel*
dtype0*
_output_shapes

:


?
u_org_fc/kernel/Adam
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_org_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_org_fc/kernel/Adam/AssignAssignu_org_fc/kernel/Adam&u_org_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_org_fc/kernel/Adam/readIdentityu_org_fc/kernel/Adam*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes

:


?
(u_org_fc/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_org_fc/kernel*
dtype0*
_output_shapes

:


?
u_org_fc/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_org_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_org_fc/kernel/Adam_1/AssignAssignu_org_fc/kernel/Adam_1(u_org_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_org_fc/kernel/Adam_1/readIdentityu_org_fc/kernel/Adam_1*
T0*"
_class
loc:@u_org_fc/kernel*
_output_shapes

:


?
$u_org_fc/bias/Adam/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_org_fc/bias*
dtype0*
_output_shapes
:

?
u_org_fc/bias/Adam
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_org_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_org_fc/bias/Adam/AssignAssignu_org_fc/bias/Adam$u_org_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

~
u_org_fc/bias/Adam/readIdentityu_org_fc/bias/Adam*
T0* 
_class
loc:@u_org_fc/bias*
_output_shapes
:

?
&u_org_fc/bias/Adam_1/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_org_fc/bias*
dtype0*
_output_shapes
:

?
u_org_fc/bias/Adam_1
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_org_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_org_fc/bias/Adam_1/AssignAssignu_org_fc/bias/Adam_1&u_org_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

?
u_org_fc/bias/Adam_1/readIdentityu_org_fc/bias/Adam_1*
T0* 
_class
loc:@u_org_fc/bias*
_output_shapes
:

?
'u_seat_fc/kernel/Adam/Initializer/zerosConst*
valueB

*    *#
_class
loc:@u_seat_fc/kernel*
dtype0*
_output_shapes

:


?
u_seat_fc/kernel/Adam
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_seat_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_seat_fc/kernel/Adam/AssignAssignu_seat_fc/kernel/Adam'u_seat_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_seat_fc/kernel/Adam/readIdentityu_seat_fc/kernel/Adam*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes

:


?
)u_seat_fc/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *#
_class
loc:@u_seat_fc/kernel*
dtype0*
_output_shapes

:


?
u_seat_fc/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *#
_class
loc:@u_seat_fc/kernel*
dtype0*
	container *
_output_shapes

:


?
u_seat_fc/kernel/Adam_1/AssignAssignu_seat_fc/kernel/Adam_1)u_seat_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
u_seat_fc/kernel/Adam_1/readIdentityu_seat_fc/kernel/Adam_1*
T0*#
_class
loc:@u_seat_fc/kernel*
_output_shapes

:


?
%u_seat_fc/bias/Adam/Initializer/zerosConst*
valueB
*    *!
_class
loc:@u_seat_fc/bias*
dtype0*
_output_shapes
:

?
u_seat_fc/bias/Adam
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_seat_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_seat_fc/bias/Adam/AssignAssignu_seat_fc/bias/Adam%u_seat_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

?
u_seat_fc/bias/Adam/readIdentityu_seat_fc/bias/Adam*
T0*!
_class
loc:@u_seat_fc/bias*
_output_shapes
:

?
'u_seat_fc/bias/Adam_1/Initializer/zerosConst*
valueB
*    *!
_class
loc:@u_seat_fc/bias*
dtype0*
_output_shapes
:

?
u_seat_fc/bias/Adam_1
VariableV2*
shape:
*
shared_name *!
_class
loc:@u_seat_fc/bias*
dtype0*
	container *
_output_shapes
:

?
u_seat_fc/bias/Adam_1/AssignAssignu_seat_fc/bias/Adam_1'u_seat_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

?
u_seat_fc/bias/Adam_1/readIdentityu_seat_fc/bias/Adam_1*
T0*!
_class
loc:@u_seat_fc/bias*
_output_shapes
:

?
&u_pos_id/kernel/Adam/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_pos_id/kernel*
dtype0*
_output_shapes

:


?
u_pos_id/kernel/Adam
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_pos_id/kernel*
dtype0*
	container *
_output_shapes

:


?
u_pos_id/kernel/Adam/AssignAssignu_pos_id/kernel/Adam&u_pos_id/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


?
u_pos_id/kernel/Adam/readIdentityu_pos_id/kernel/Adam*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes

:


?
(u_pos_id/kernel/Adam_1/Initializer/zerosConst*
valueB

*    *"
_class
loc:@u_pos_id/kernel*
dtype0*
_output_shapes

:


?
u_pos_id/kernel/Adam_1
VariableV2*
shape
:

*
shared_name *"
_class
loc:@u_pos_id/kernel*
dtype0*
	container *
_output_shapes

:


?
u_pos_id/kernel/Adam_1/AssignAssignu_pos_id/kernel/Adam_1(u_pos_id/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


?
u_pos_id/kernel/Adam_1/readIdentityu_pos_id/kernel/Adam_1*
T0*"
_class
loc:@u_pos_id/kernel*
_output_shapes

:


?
$u_pos_id/bias/Adam/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_pos_id/bias*
dtype0*
_output_shapes
:

?
u_pos_id/bias/Adam
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_pos_id/bias*
dtype0*
	container *
_output_shapes
:

?
u_pos_id/bias/Adam/AssignAssignu_pos_id/bias/Adam$u_pos_id/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

~
u_pos_id/bias/Adam/readIdentityu_pos_id/bias/Adam*
T0* 
_class
loc:@u_pos_id/bias*
_output_shapes
:

?
&u_pos_id/bias/Adam_1/Initializer/zerosConst*
valueB
*    * 
_class
loc:@u_pos_id/bias*
dtype0*
_output_shapes
:

?
u_pos_id/bias/Adam_1
VariableV2*
shape:
*
shared_name * 
_class
loc:@u_pos_id/bias*
dtype0*
	container *
_output_shapes
:

?
u_pos_id/bias/Adam_1/AssignAssignu_pos_id/bias/Adam_1&u_pos_id/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

?
u_pos_id/bias/Adam_1/readIdentityu_pos_id/bias/Adam_1*
T0* 
_class
loc:@u_pos_id/bias*
_output_shapes
:

?
;u_concated_fc/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *'
_class
loc:@u_concated_fc/kernel*
dtype0*
_output_shapes
:
?
1u_concated_fc/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@u_concated_fc/kernel*
dtype0*
_output_shapes
: 
?
+u_concated_fc/kernel/Adam/Initializer/zerosFill;u_concated_fc/kernel/Adam/Initializer/zeros/shape_as_tensor1u_concated_fc/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
u_concated_fc/kernel/Adam
VariableV2*
shape
:<@*
shared_name *'
_class
loc:@u_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:<@
?
 u_concated_fc/kernel/Adam/AssignAssignu_concated_fc/kernel/Adam+u_concated_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
u_concated_fc/kernel/Adam/readIdentityu_concated_fc/kernel/Adam*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
=u_concated_fc/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *'
_class
loc:@u_concated_fc/kernel*
dtype0*
_output_shapes
:
?
3u_concated_fc/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@u_concated_fc/kernel*
dtype0*
_output_shapes
: 
?
-u_concated_fc/kernel/Adam_1/Initializer/zerosFill=u_concated_fc/kernel/Adam_1/Initializer/zeros/shape_as_tensor3u_concated_fc/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
u_concated_fc/kernel/Adam_1
VariableV2*
shape
:<@*
shared_name *'
_class
loc:@u_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:<@
?
"u_concated_fc/kernel/Adam_1/AssignAssignu_concated_fc/kernel/Adam_1-u_concated_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
 u_concated_fc/kernel/Adam_1/readIdentityu_concated_fc/kernel/Adam_1*
T0*'
_class
loc:@u_concated_fc/kernel*
_output_shapes

:<@
?
)u_concated_fc/bias/Adam/Initializer/zerosConst*
valueB@*    *%
_class
loc:@u_concated_fc/bias*
dtype0*
_output_shapes
:@
?
u_concated_fc/bias/Adam
VariableV2*
shape:@*
shared_name *%
_class
loc:@u_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
u_concated_fc/bias/Adam/AssignAssignu_concated_fc/bias/Adam)u_concated_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
u_concated_fc/bias/Adam/readIdentityu_concated_fc/bias/Adam*
T0*%
_class
loc:@u_concated_fc/bias*
_output_shapes
:@
?
+u_concated_fc/bias/Adam_1/Initializer/zerosConst*
valueB@*    *%
_class
loc:@u_concated_fc/bias*
dtype0*
_output_shapes
:@
?
u_concated_fc/bias/Adam_1
VariableV2*
shape:@*
shared_name *%
_class
loc:@u_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
 u_concated_fc/bias/Adam_1/AssignAssignu_concated_fc/bias/Adam_1+u_concated_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
u_concated_fc/bias/Adam_1/readIdentityu_concated_fc/bias/Adam_1*
T0*%
_class
loc:@u_concated_fc/bias*
_output_shapes
:@
?
>item_class_embedding/i_class_emb_matrix/Adam/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
dtype0*
_output_shapes

:
?
,item_class_embedding/i_class_emb_matrix/Adam
VariableV2*
shape
:*
shared_name *:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
dtype0*
	container *
_output_shapes

:
?
3item_class_embedding/i_class_emb_matrix/Adam/AssignAssign,item_class_embedding/i_class_emb_matrix/Adam>item_class_embedding/i_class_emb_matrix/Adam/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
1item_class_embedding/i_class_emb_matrix/Adam/readIdentity,item_class_embedding/i_class_emb_matrix/Adam*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
@item_class_embedding/i_class_emb_matrix/Adam_1/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
dtype0*
_output_shapes

:
?
.item_class_embedding/i_class_emb_matrix/Adam_1
VariableV2*
shape
:*
shared_name *:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
dtype0*
	container *
_output_shapes

:
?
5item_class_embedding/i_class_emb_matrix/Adam_1/AssignAssign.item_class_embedding/i_class_emb_matrix/Adam_1@item_class_embedding/i_class_emb_matrix/Adam_1/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
3item_class_embedding/i_class_emb_matrix/Adam_1/readIdentity.item_class_embedding/i_class_emb_matrix/Adam_1*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
(i_class_fc/kernel/Adam/Initializer/zerosConst*
valueB*    *$
_class
loc:@i_class_fc/kernel*
dtype0*
_output_shapes

:
?
i_class_fc/kernel/Adam
VariableV2*
shape
:*
shared_name *$
_class
loc:@i_class_fc/kernel*
dtype0*
	container *
_output_shapes

:
?
i_class_fc/kernel/Adam/AssignAssigni_class_fc/kernel/Adam(i_class_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
i_class_fc/kernel/Adam/readIdentityi_class_fc/kernel/Adam*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes

:
?
*i_class_fc/kernel/Adam_1/Initializer/zerosConst*
valueB*    *$
_class
loc:@i_class_fc/kernel*
dtype0*
_output_shapes

:
?
i_class_fc/kernel/Adam_1
VariableV2*
shape
:*
shared_name *$
_class
loc:@i_class_fc/kernel*
dtype0*
	container *
_output_shapes

:
?
i_class_fc/kernel/Adam_1/AssignAssigni_class_fc/kernel/Adam_1*i_class_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
i_class_fc/kernel/Adam_1/readIdentityi_class_fc/kernel/Adam_1*
T0*$
_class
loc:@i_class_fc/kernel*
_output_shapes

:
?
&i_class_fc/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@i_class_fc/bias*
dtype0*
_output_shapes
:
?
i_class_fc/bias/Adam
VariableV2*
shape:*
shared_name *"
_class
loc:@i_class_fc/bias*
dtype0*
	container *
_output_shapes
:
?
i_class_fc/bias/Adam/AssignAssigni_class_fc/bias/Adam&i_class_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
?
i_class_fc/bias/Adam/readIdentityi_class_fc/bias/Adam*
T0*"
_class
loc:@i_class_fc/bias*
_output_shapes
:
?
(i_class_fc/bias/Adam_1/Initializer/zerosConst*
valueB*    *"
_class
loc:@i_class_fc/bias*
dtype0*
_output_shapes
:
?
i_class_fc/bias/Adam_1
VariableV2*
shape:*
shared_name *"
_class
loc:@i_class_fc/bias*
dtype0*
	container *
_output_shapes
:
?
i_class_fc/bias/Adam_1/AssignAssigni_class_fc/bias/Adam_1(i_class_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
?
i_class_fc/bias/Adam_1/readIdentityi_class_fc/bias/Adam_1*
T0*"
_class
loc:@i_class_fc/bias*
_output_shapes
:
?
;i_concated_fc/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   @   *'
_class
loc:@i_concated_fc/kernel*
dtype0*
_output_shapes
:
?
1i_concated_fc/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@i_concated_fc/kernel*
dtype0*
_output_shapes
: 
?
+i_concated_fc/kernel/Adam/Initializer/zerosFill;i_concated_fc/kernel/Adam/Initializer/zeros/shape_as_tensor1i_concated_fc/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
i_concated_fc/kernel/Adam
VariableV2*
shape
:@*
shared_name *'
_class
loc:@i_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:@
?
 i_concated_fc/kernel/Adam/AssignAssigni_concated_fc/kernel/Adam+i_concated_fc/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
i_concated_fc/kernel/Adam/readIdentityi_concated_fc/kernel/Adam*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
=i_concated_fc/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   @   *'
_class
loc:@i_concated_fc/kernel*
dtype0*
_output_shapes
:
?
3i_concated_fc/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@i_concated_fc/kernel*
dtype0*
_output_shapes
: 
?
-i_concated_fc/kernel/Adam_1/Initializer/zerosFill=i_concated_fc/kernel/Adam_1/Initializer/zeros/shape_as_tensor3i_concated_fc/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
i_concated_fc/kernel/Adam_1
VariableV2*
shape
:@*
shared_name *'
_class
loc:@i_concated_fc/kernel*
dtype0*
	container *
_output_shapes

:@
?
"i_concated_fc/kernel/Adam_1/AssignAssigni_concated_fc/kernel/Adam_1-i_concated_fc/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
 i_concated_fc/kernel/Adam_1/readIdentityi_concated_fc/kernel/Adam_1*
T0*'
_class
loc:@i_concated_fc/kernel*
_output_shapes

:@
?
)i_concated_fc/bias/Adam/Initializer/zerosConst*
valueB@*    *%
_class
loc:@i_concated_fc/bias*
dtype0*
_output_shapes
:@
?
i_concated_fc/bias/Adam
VariableV2*
shape:@*
shared_name *%
_class
loc:@i_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
i_concated_fc/bias/Adam/AssignAssigni_concated_fc/bias/Adam)i_concated_fc/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
i_concated_fc/bias/Adam/readIdentityi_concated_fc/bias/Adam*
T0*%
_class
loc:@i_concated_fc/bias*
_output_shapes
:@
?
+i_concated_fc/bias/Adam_1/Initializer/zerosConst*
valueB@*    *%
_class
loc:@i_concated_fc/bias*
dtype0*
_output_shapes
:@
?
i_concated_fc/bias/Adam_1
VariableV2*
shape:@*
shared_name *%
_class
loc:@i_concated_fc/bias*
dtype0*
	container *
_output_shapes
:@
?
 i_concated_fc/bias/Adam_1/AssignAssigni_concated_fc/bias/Adam_1+i_concated_fc/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
i_concated_fc/bias/Adam_1/readIdentityi_concated_fc/bias/Adam_1*
T0*%
_class
loc:@i_concated_fc/bias*
_output_shapes
:@
?
4expert_weight/Adam/Initializer/zeros/shape_as_tensorConst*!
valueB"?         * 
_class
loc:@expert_weight*
dtype0*
_output_shapes
:
?
*expert_weight/Adam/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@expert_weight*
dtype0*
_output_shapes
: 
?
$expert_weight/Adam/Initializer/zerosFill4expert_weight/Adam/Initializer/zeros/shape_as_tensor*expert_weight/Adam/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
expert_weight/Adam
VariableV2*
shape:?*
shared_name * 
_class
loc:@expert_weight*
dtype0*
	container *#
_output_shapes
:?
?
expert_weight/Adam/AssignAssignexpert_weight/Adam$expert_weight/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
?
expert_weight/Adam/readIdentityexpert_weight/Adam*
T0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
6expert_weight/Adam_1/Initializer/zeros/shape_as_tensorConst*!
valueB"?         * 
_class
loc:@expert_weight*
dtype0*
_output_shapes
:
?
,expert_weight/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    * 
_class
loc:@expert_weight*
dtype0*
_output_shapes
: 
?
&expert_weight/Adam_1/Initializer/zerosFill6expert_weight/Adam_1/Initializer/zeros/shape_as_tensor,expert_weight/Adam_1/Initializer/zeros/Const*
T0*

index_type0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
expert_weight/Adam_1
VariableV2*
shape:?*
shared_name * 
_class
loc:@expert_weight*
dtype0*
	container *#
_output_shapes
:?
?
expert_weight/Adam_1/AssignAssignexpert_weight/Adam_1&expert_weight/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
?
expert_weight/Adam_1/readIdentityexpert_weight/Adam_1*
T0* 
_class
loc:@expert_weight*#
_output_shapes
:?
?
"expert_bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@expert_bias*
dtype0*
_output_shapes
:
?
expert_bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@expert_bias*
dtype0*
	container *
_output_shapes
:
?
expert_bias/Adam/AssignAssignexpert_bias/Adam"expert_bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
x
expert_bias/Adam/readIdentityexpert_bias/Adam*
T0*
_class
loc:@expert_bias*
_output_shapes
:
?
$expert_bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@expert_bias*
dtype0*
_output_shapes
:
?
expert_bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@expert_bias*
dtype0*
	container *
_output_shapes
:
?
expert_bias/Adam_1/AssignAssignexpert_bias/Adam_1$expert_bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
|
expert_bias/Adam_1/readIdentityexpert_bias/Adam_1*
T0*
_class
loc:@expert_bias*
_output_shapes
:
?
#gate1_weight/Adam/Initializer/zerosConst*
valueB	?*    *
_class
loc:@gate1_weight*
dtype0*
_output_shapes
:	?
?
gate1_weight/Adam
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate1_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate1_weight/Adam/AssignAssigngate1_weight/Adam#gate1_weight/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
?
gate1_weight/Adam/readIdentitygate1_weight/Adam*
T0*
_class
loc:@gate1_weight*
_output_shapes
:	?
?
%gate1_weight/Adam_1/Initializer/zerosConst*
valueB	?*    *
_class
loc:@gate1_weight*
dtype0*
_output_shapes
:	?
?
gate1_weight/Adam_1
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate1_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate1_weight/Adam_1/AssignAssigngate1_weight/Adam_1%gate1_weight/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
?
gate1_weight/Adam_1/readIdentitygate1_weight/Adam_1*
T0*
_class
loc:@gate1_weight*
_output_shapes
:	?
?
!gate1_bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@gate1_bias*
dtype0*
_output_shapes
:
?
gate1_bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@gate1_bias*
dtype0*
	container *
_output_shapes
:
?
gate1_bias/Adam/AssignAssigngate1_bias/Adam!gate1_bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
u
gate1_bias/Adam/readIdentitygate1_bias/Adam*
T0*
_class
loc:@gate1_bias*
_output_shapes
:
?
#gate1_bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@gate1_bias*
dtype0*
_output_shapes
:
?
gate1_bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@gate1_bias*
dtype0*
	container *
_output_shapes
:
?
gate1_bias/Adam_1/AssignAssigngate1_bias/Adam_1#gate1_bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
y
gate1_bias/Adam_1/readIdentitygate1_bias/Adam_1*
T0*
_class
loc:@gate1_bias*
_output_shapes
:
?
#gate2_weight/Adam/Initializer/zerosConst*
valueB	?*    *
_class
loc:@gate2_weight*
dtype0*
_output_shapes
:	?
?
gate2_weight/Adam
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate2_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate2_weight/Adam/AssignAssigngate2_weight/Adam#gate2_weight/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
?
gate2_weight/Adam/readIdentitygate2_weight/Adam*
T0*
_class
loc:@gate2_weight*
_output_shapes
:	?
?
%gate2_weight/Adam_1/Initializer/zerosConst*
valueB	?*    *
_class
loc:@gate2_weight*
dtype0*
_output_shapes
:	?
?
gate2_weight/Adam_1
VariableV2*
shape:	?*
shared_name *
_class
loc:@gate2_weight*
dtype0*
	container *
_output_shapes
:	?
?
gate2_weight/Adam_1/AssignAssigngate2_weight/Adam_1%gate2_weight/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
?
gate2_weight/Adam_1/readIdentitygate2_weight/Adam_1*
T0*
_class
loc:@gate2_weight*
_output_shapes
:	?
?
!gate2_bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@gate2_bias*
dtype0*
_output_shapes
:
?
gate2_bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@gate2_bias*
dtype0*
	container *
_output_shapes
:
?
gate2_bias/Adam/AssignAssigngate2_bias/Adam!gate2_bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
u
gate2_bias/Adam/readIdentitygate2_bias/Adam*
T0*
_class
loc:@gate2_bias*
_output_shapes
:
?
#gate2_bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@gate2_bias*
dtype0*
_output_shapes
:
?
gate2_bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@gate2_bias*
dtype0*
	container *
_output_shapes
:
?
gate2_bias/Adam_1/AssignAssigngate2_bias/Adam_1#gate2_bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
y
gate2_bias/Adam_1/readIdentitygate2_bias/Adam_1*
T0*
_class
loc:@gate2_bias*
_output_shapes
:
?
#dense/kernel/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
?
dense/kernel/Adam
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
?
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
%dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
?
dense/kernel/Adam_1
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
?
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
?
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
?
!dense/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
?
dense/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
:
?
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
u
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
#dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
?
dense/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
:
?
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:
?
%dense_1/kernel/Adam/Initializer/zerosConst*
valueB *    *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

: 
?
dense_1/kernel/Adam
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
?
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
?
'dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB *    *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

: 
?
dense_1/kernel/Adam_1
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
?
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

: 
?
#dense_1/bias/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
?
dense_1/bias/Adam
VariableV2*
shape: *
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container *
_output_shapes
: 
?
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
{
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_class
loc:@dense_1/bias*
_output_shapes
: 
?
%dense_1/bias/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
?
dense_1/bias/Adam_1
VariableV2*
shape: *
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container *
_output_shapes
: 
?
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 

dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
: 
?
%dense_2/kernel/Adam/Initializer/zerosConst*
valueB *    *!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes

: 
?
dense_2/kernel/Adam
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adam%dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
?
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
?
'dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB *    *!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes

: 
?
dense_2/kernel/Adam_1
VariableV2*
shape
: *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container *
_output_shapes

: 
?
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1'dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
?
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
?
#dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
?
dense_2/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container *
_output_shapes
:
?
dense_2/bias/Adam/AssignAssigndense_2/bias/Adam#dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
?
%dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
?
dense_2/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container *
_output_shapes
:
?
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1%dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
?
%dense_3/kernel/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes

:
?
dense_3/kernel/Adam
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_3/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_3/kernel/Adam/AssignAssigndense_3/kernel/Adam%dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
?
dense_3/kernel/Adam/readIdentitydense_3/kernel/Adam*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
?
'dense_3/kernel/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes

:
?
dense_3/kernel/Adam_1
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_3/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_3/kernel/Adam_1/AssignAssigndense_3/kernel/Adam_1'dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
?
dense_3/kernel/Adam_1/readIdentitydense_3/kernel/Adam_1*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
?
#dense_3/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:
?
dense_3/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense_3/bias*
dtype0*
	container *
_output_shapes
:
?
dense_3/bias/Adam/AssignAssigndense_3/bias/Adam#dense_3/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
{
dense_3/bias/Adam/readIdentitydense_3/bias/Adam*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
?
%dense_3/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:
?
dense_3/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense_3/bias*
dtype0*
	container *
_output_shapes
:
?
dense_3/bias/Adam_1/AssignAssigndense_3/bias/Adam_1%dense_3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:

dense_3/bias/Adam_1/readIdentitydense_3/bias/Adam_1*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
?
%dense_5/kernel/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:
?
dense_5/kernel/Adam
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_5/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_5/kernel/Adam/AssignAssigndense_5/kernel/Adam%dense_5/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
?
dense_5/kernel/Adam/readIdentitydense_5/kernel/Adam*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
?
'dense_5/kernel/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:
?
dense_5/kernel/Adam_1
VariableV2*
shape
:*
shared_name *!
_class
loc:@dense_5/kernel*
dtype0*
	container *
_output_shapes

:
?
dense_5/kernel/Adam_1/AssignAssigndense_5/kernel/Adam_1'dense_5/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
?
dense_5/kernel/Adam_1/readIdentitydense_5/kernel/Adam_1*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
?
#dense_5/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:
?
dense_5/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense_5/bias*
dtype0*
	container *
_output_shapes
:
?
dense_5/bias/Adam/AssignAssigndense_5/bias/Adam#dense_5/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
{
dense_5/bias/Adam/readIdentitydense_5/bias/Adam*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:
?
%dense_5/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:
?
dense_5/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense_5/bias*
dtype0*
	container *
_output_shapes
:
?
dense_5/bias/Adam_1/AssignAssigndense_5/bias/Adam_1%dense_5/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:

dense_5/bias/Adam_1/readIdentitydense_5/bias/Adam_1*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o?:*
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
 *w??*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
3Adam/update_user_embedding/u_type_emb_matrix/UniqueUnique8gradients/user_embedding/u_type_emb_layer_grad/Reshape_1*
out_idx0*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*2
_output_shapes 
:?????????:?????????
?
2Adam/update_user_embedding/u_type_emb_matrix/ShapeShape3Adam/update_user_embedding/u_type_emb_matrix/Unique*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
out_type0*
_output_shapes
:
?
@Adam/update_user_embedding/u_type_emb_matrix/strided_slice/stackConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
BAdam/update_user_embedding/u_type_emb_matrix/strided_slice/stack_1Const*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
BAdam/update_user_embedding/u_type_emb_matrix/strided_slice/stack_2Const*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
:Adam/update_user_embedding/u_type_emb_matrix/strided_sliceStridedSlice2Adam/update_user_embedding/u_type_emb_matrix/Shape@Adam/update_user_embedding/u_type_emb_matrix/strided_slice/stackBAdam/update_user_embedding/u_type_emb_matrix/strided_slice/stack_1BAdam/update_user_embedding/u_type_emb_matrix/strided_slice/stack_2*
T0*
Index0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
?Adam/update_user_embedding/u_type_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum6gradients/user_embedding/u_type_emb_layer_grad/Reshape5Adam/update_user_embedding/u_type_emb_matrix/Unique:1:Adam/update_user_embedding/u_type_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_type_emb_matrix/sub/xConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
0Adam/update_user_embedding/u_type_emb_matrix/subSub2Adam/update_user_embedding/u_type_emb_matrix/sub/xbeta2_power/read*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_type_emb_matrix/SqrtSqrt0Adam/update_user_embedding/u_type_emb_matrix/sub*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_type_emb_matrix/mulMulAdam/learning_rate1Adam/update_user_embedding/u_type_emb_matrix/Sqrt*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_type_emb_matrix/sub_1/xConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_type_emb_matrix/sub_1Sub4Adam/update_user_embedding/u_type_emb_matrix/sub_1/xbeta1_power/read*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_type_emb_matrix/truedivRealDiv0Adam/update_user_embedding/u_type_emb_matrix/mul2Adam/update_user_embedding/u_type_emb_matrix/sub_1*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_type_emb_matrix/sub_2/xConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_type_emb_matrix/sub_2Sub4Adam/update_user_embedding/u_type_emb_matrix/sub_2/x
Adam/beta1*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
2Adam/update_user_embedding/u_type_emb_matrix/mul_1Mul?Adam/update_user_embedding/u_type_emb_matrix/UnsortedSegmentSum2Adam/update_user_embedding/u_type_emb_matrix/sub_2*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_type_emb_matrix/mul_2Mul*user_embedding/u_type_emb_matrix/Adam/read
Adam/beta1*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
3Adam/update_user_embedding/u_type_emb_matrix/AssignAssign%user_embedding/u_type_emb_matrix/Adam2Adam/update_user_embedding/u_type_emb_matrix/mul_2*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
7Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd
ScatterAdd%user_embedding/u_type_emb_matrix/Adam3Adam/update_user_embedding/u_type_emb_matrix/Unique2Adam/update_user_embedding/u_type_emb_matrix/mul_14^Adam/update_user_embedding/u_type_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_type_emb_matrix/mul_3Mul?Adam/update_user_embedding/u_type_emb_matrix/UnsortedSegmentSum?Adam/update_user_embedding/u_type_emb_matrix/UnsortedSegmentSum*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*'
_output_shapes
:?????????

?
4Adam/update_user_embedding/u_type_emb_matrix/sub_3/xConst*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_type_emb_matrix/sub_3Sub4Adam/update_user_embedding/u_type_emb_matrix/sub_3/x
Adam/beta2*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes
: 
?
2Adam/update_user_embedding/u_type_emb_matrix/mul_4Mul2Adam/update_user_embedding/u_type_emb_matrix/mul_32Adam/update_user_embedding/u_type_emb_matrix/sub_3*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_type_emb_matrix/mul_5Mul,user_embedding/u_type_emb_matrix/Adam_1/read
Adam/beta2*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_type_emb_matrix/Assign_1Assign'user_embedding/u_type_emb_matrix/Adam_12Adam/update_user_embedding/u_type_emb_matrix/mul_5*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
9Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd_1
ScatterAdd'user_embedding/u_type_emb_matrix/Adam_13Adam/update_user_embedding/u_type_emb_matrix/Unique2Adam/update_user_embedding/u_type_emb_matrix/mul_46^Adam/update_user_embedding/u_type_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
3Adam/update_user_embedding/u_type_emb_matrix/Sqrt_1Sqrt9Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd_1*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_type_emb_matrix/mul_6Mul4Adam/update_user_embedding/u_type_emb_matrix/truediv7Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
0Adam/update_user_embedding/u_type_emb_matrix/addAdd3Adam/update_user_embedding/u_type_emb_matrix/Sqrt_1Adam/epsilon*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_type_emb_matrix/truediv_1RealDiv2Adam/update_user_embedding/u_type_emb_matrix/mul_60Adam/update_user_embedding/u_type_emb_matrix/add*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_type_emb_matrix/AssignSub	AssignSub user_embedding/u_type_emb_matrix6Adam/update_user_embedding/u_type_emb_matrix/truediv_1*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
_output_shapes

:

?
7Adam/update_user_embedding/u_type_emb_matrix/group_depsNoOp7^Adam/update_user_embedding/u_type_emb_matrix/AssignSub8^Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd:^Adam/update_user_embedding/u_type_emb_matrix/ScatterAdd_1*3
_class)
'%loc:@user_embedding/u_type_emb_matrix
?
2Adam/update_user_embedding/u_age_emn_matrix/UniqueUnique7gradients/user_embedding/u_age_emb_layer_grad/Reshape_1*
out_idx0*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*2
_output_shapes 
:?????????:?????????
?
1Adam/update_user_embedding/u_age_emn_matrix/ShapeShape2Adam/update_user_embedding/u_age_emn_matrix/Unique*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
out_type0*
_output_shapes
:
?
?Adam/update_user_embedding/u_age_emn_matrix/strided_slice/stackConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB: *
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_age_emn_matrix/strided_slice/stack_1Const*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB:*
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_age_emn_matrix/strided_slice/stack_2Const*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB:*
dtype0*
_output_shapes
:
?
9Adam/update_user_embedding/u_age_emn_matrix/strided_sliceStridedSlice1Adam/update_user_embedding/u_age_emn_matrix/Shape?Adam/update_user_embedding/u_age_emn_matrix/strided_slice/stackAAdam/update_user_embedding/u_age_emn_matrix/strided_slice/stack_1AAdam/update_user_embedding/u_age_emn_matrix/strided_slice/stack_2*
T0*
Index0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
>Adam/update_user_embedding/u_age_emn_matrix/UnsortedSegmentSumUnsortedSegmentSum5gradients/user_embedding/u_age_emb_layer_grad/Reshape4Adam/update_user_embedding/u_age_emn_matrix/Unique:19Adam/update_user_embedding/u_age_emn_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_age_emn_matrix/sub/xConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
/Adam/update_user_embedding/u_age_emn_matrix/subSub1Adam/update_user_embedding/u_age_emn_matrix/sub/xbeta2_power/read*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_age_emn_matrix/SqrtSqrt/Adam/update_user_embedding/u_age_emn_matrix/sub*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
/Adam/update_user_embedding/u_age_emn_matrix/mulMulAdam/learning_rate0Adam/update_user_embedding/u_age_emn_matrix/Sqrt*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_age_emn_matrix/sub_1/xConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_age_emn_matrix/sub_1Sub3Adam/update_user_embedding/u_age_emn_matrix/sub_1/xbeta1_power/read*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_age_emn_matrix/truedivRealDiv/Adam/update_user_embedding/u_age_emn_matrix/mul1Adam/update_user_embedding/u_age_emn_matrix/sub_1*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_age_emn_matrix/sub_2/xConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_age_emn_matrix/sub_2Sub3Adam/update_user_embedding/u_age_emn_matrix/sub_2/x
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_age_emn_matrix/mul_1Mul>Adam/update_user_embedding/u_age_emn_matrix/UnsortedSegmentSum1Adam/update_user_embedding/u_age_emn_matrix/sub_2*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_age_emn_matrix/mul_2Mul)user_embedding/u_age_emn_matrix/Adam/read
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_age_emn_matrix/AssignAssign$user_embedding/u_age_emn_matrix/Adam1Adam/update_user_embedding/u_age_emn_matrix/mul_2*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
6Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd
ScatterAdd$user_embedding/u_age_emn_matrix/Adam2Adam/update_user_embedding/u_age_emn_matrix/Unique1Adam/update_user_embedding/u_age_emn_matrix/mul_13^Adam/update_user_embedding/u_age_emn_matrix/Assign*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_age_emn_matrix/mul_3Mul>Adam/update_user_embedding/u_age_emn_matrix/UnsortedSegmentSum>Adam/update_user_embedding/u_age_emn_matrix/UnsortedSegmentSum*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*'
_output_shapes
:?????????

?
3Adam/update_user_embedding/u_age_emn_matrix/sub_3/xConst*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_age_emn_matrix/sub_3Sub3Adam/update_user_embedding/u_age_emn_matrix/sub_3/x
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_age_emn_matrix/mul_4Mul1Adam/update_user_embedding/u_age_emn_matrix/mul_31Adam/update_user_embedding/u_age_emn_matrix/sub_3*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_age_emn_matrix/mul_5Mul+user_embedding/u_age_emn_matrix/Adam_1/read
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
4Adam/update_user_embedding/u_age_emn_matrix/Assign_1Assign&user_embedding/u_age_emn_matrix/Adam_11Adam/update_user_embedding/u_age_emn_matrix/mul_5*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
8Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd_1
ScatterAdd&user_embedding/u_age_emn_matrix/Adam_12Adam/update_user_embedding/u_age_emn_matrix/Unique1Adam/update_user_embedding/u_age_emn_matrix/mul_45^Adam/update_user_embedding/u_age_emn_matrix/Assign_1*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_age_emn_matrix/Sqrt_1Sqrt8Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd_1*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_age_emn_matrix/mul_6Mul3Adam/update_user_embedding/u_age_emn_matrix/truediv6Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
/Adam/update_user_embedding/u_age_emn_matrix/addAdd2Adam/update_user_embedding/u_age_emn_matrix/Sqrt_1Adam/epsilon*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_age_emn_matrix/truediv_1RealDiv1Adam/update_user_embedding/u_age_emn_matrix/mul_6/Adam/update_user_embedding/u_age_emn_matrix/add*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_age_emn_matrix/AssignSub	AssignSubuser_embedding/u_age_emn_matrix5Adam/update_user_embedding/u_age_emn_matrix/truediv_1*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_age_emn_matrix/group_depsNoOp6^Adam/update_user_embedding/u_age_emn_matrix/AssignSub7^Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd9^Adam/update_user_embedding/u_age_emn_matrix/ScatterAdd_1*2
_class(
&$loc:@user_embedding/u_age_emn_matrix
?
2Adam/update_user_embedding/u_sex_emb_matrix/UniqueUnique7gradients/user_embedding/u_sex_emb_layer_grad/Reshape_1*
out_idx0*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*2
_output_shapes 
:?????????:?????????
?
1Adam/update_user_embedding/u_sex_emb_matrix/ShapeShape2Adam/update_user_embedding/u_sex_emb_matrix/Unique*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
out_type0*
_output_shapes
:
?
?Adam/update_user_embedding/u_sex_emb_matrix/strided_slice/stackConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_sex_emb_matrix/strided_slice/stack_1Const*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_sex_emb_matrix/strided_slice/stack_2Const*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
9Adam/update_user_embedding/u_sex_emb_matrix/strided_sliceStridedSlice1Adam/update_user_embedding/u_sex_emb_matrix/Shape?Adam/update_user_embedding/u_sex_emb_matrix/strided_slice/stackAAdam/update_user_embedding/u_sex_emb_matrix/strided_slice/stack_1AAdam/update_user_embedding/u_sex_emb_matrix/strided_slice/stack_2*
T0*
Index0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
>Adam/update_user_embedding/u_sex_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum5gradients/user_embedding/u_sex_emb_layer_grad/Reshape4Adam/update_user_embedding/u_sex_emb_matrix/Unique:19Adam/update_user_embedding/u_sex_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_sex_emb_matrix/sub/xConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
/Adam/update_user_embedding/u_sex_emb_matrix/subSub1Adam/update_user_embedding/u_sex_emb_matrix/sub/xbeta2_power/read*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_sex_emb_matrix/SqrtSqrt/Adam/update_user_embedding/u_sex_emb_matrix/sub*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
/Adam/update_user_embedding/u_sex_emb_matrix/mulMulAdam/learning_rate0Adam/update_user_embedding/u_sex_emb_matrix/Sqrt*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_sex_emb_matrix/sub_1/xConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_sex_emb_matrix/sub_1Sub3Adam/update_user_embedding/u_sex_emb_matrix/sub_1/xbeta1_power/read*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_sex_emb_matrix/truedivRealDiv/Adam/update_user_embedding/u_sex_emb_matrix/mul1Adam/update_user_embedding/u_sex_emb_matrix/sub_1*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_sex_emb_matrix/sub_2/xConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_sex_emb_matrix/sub_2Sub3Adam/update_user_embedding/u_sex_emb_matrix/sub_2/x
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_1Mul>Adam/update_user_embedding/u_sex_emb_matrix/UnsortedSegmentSum1Adam/update_user_embedding/u_sex_emb_matrix/sub_2*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_2Mul)user_embedding/u_sex_emb_matrix/Adam/read
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_sex_emb_matrix/AssignAssign$user_embedding/u_sex_emb_matrix/Adam1Adam/update_user_embedding/u_sex_emb_matrix/mul_2*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
6Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd
ScatterAdd$user_embedding/u_sex_emb_matrix/Adam2Adam/update_user_embedding/u_sex_emb_matrix/Unique1Adam/update_user_embedding/u_sex_emb_matrix/mul_13^Adam/update_user_embedding/u_sex_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_3Mul>Adam/update_user_embedding/u_sex_emb_matrix/UnsortedSegmentSum>Adam/update_user_embedding/u_sex_emb_matrix/UnsortedSegmentSum*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*'
_output_shapes
:?????????

?
3Adam/update_user_embedding/u_sex_emb_matrix/sub_3/xConst*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_sex_emb_matrix/sub_3Sub3Adam/update_user_embedding/u_sex_emb_matrix/sub_3/x
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_4Mul1Adam/update_user_embedding/u_sex_emb_matrix/mul_31Adam/update_user_embedding/u_sex_emb_matrix/sub_3*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_5Mul+user_embedding/u_sex_emb_matrix/Adam_1/read
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
4Adam/update_user_embedding/u_sex_emb_matrix/Assign_1Assign&user_embedding/u_sex_emb_matrix/Adam_11Adam/update_user_embedding/u_sex_emb_matrix/mul_5*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
8Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd_1
ScatterAdd&user_embedding/u_sex_emb_matrix/Adam_12Adam/update_user_embedding/u_sex_emb_matrix/Unique1Adam/update_user_embedding/u_sex_emb_matrix/mul_45^Adam/update_user_embedding/u_sex_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_sex_emb_matrix/Sqrt_1Sqrt8Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd_1*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_sex_emb_matrix/mul_6Mul3Adam/update_user_embedding/u_sex_emb_matrix/truediv6Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
/Adam/update_user_embedding/u_sex_emb_matrix/addAdd2Adam/update_user_embedding/u_sex_emb_matrix/Sqrt_1Adam/epsilon*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_sex_emb_matrix/truediv_1RealDiv1Adam/update_user_embedding/u_sex_emb_matrix/mul_6/Adam/update_user_embedding/u_sex_emb_matrix/add*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_sex_emb_matrix/AssignSub	AssignSubuser_embedding/u_sex_emb_matrix5Adam/update_user_embedding/u_sex_emb_matrix/truediv_1*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_sex_emb_matrix/group_depsNoOp6^Adam/update_user_embedding/u_sex_emb_matrix/AssignSub7^Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd9^Adam/update_user_embedding/u_sex_emb_matrix/ScatterAdd_1*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix
?
2Adam/update_user_embedding/u_org_emb_matrix/UniqueUnique7gradients/user_embedding/u_org_emb_layer_grad/Reshape_1*
out_idx0*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*2
_output_shapes 
:?????????:?????????
?
1Adam/update_user_embedding/u_org_emb_matrix/ShapeShape2Adam/update_user_embedding/u_org_emb_matrix/Unique*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
out_type0*
_output_shapes
:
?
?Adam/update_user_embedding/u_org_emb_matrix/strided_slice/stackConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_org_emb_matrix/strided_slice/stack_1Const*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_org_emb_matrix/strided_slice/stack_2Const*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
9Adam/update_user_embedding/u_org_emb_matrix/strided_sliceStridedSlice1Adam/update_user_embedding/u_org_emb_matrix/Shape?Adam/update_user_embedding/u_org_emb_matrix/strided_slice/stackAAdam/update_user_embedding/u_org_emb_matrix/strided_slice/stack_1AAdam/update_user_embedding/u_org_emb_matrix/strided_slice/stack_2*
T0*
Index0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
>Adam/update_user_embedding/u_org_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum5gradients/user_embedding/u_org_emb_layer_grad/Reshape4Adam/update_user_embedding/u_org_emb_matrix/Unique:19Adam/update_user_embedding/u_org_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_org_emb_matrix/sub/xConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
/Adam/update_user_embedding/u_org_emb_matrix/subSub1Adam/update_user_embedding/u_org_emb_matrix/sub/xbeta2_power/read*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_org_emb_matrix/SqrtSqrt/Adam/update_user_embedding/u_org_emb_matrix/sub*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
/Adam/update_user_embedding/u_org_emb_matrix/mulMulAdam/learning_rate0Adam/update_user_embedding/u_org_emb_matrix/Sqrt*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_org_emb_matrix/sub_1/xConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_org_emb_matrix/sub_1Sub3Adam/update_user_embedding/u_org_emb_matrix/sub_1/xbeta1_power/read*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_org_emb_matrix/truedivRealDiv/Adam/update_user_embedding/u_org_emb_matrix/mul1Adam/update_user_embedding/u_org_emb_matrix/sub_1*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_org_emb_matrix/sub_2/xConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_org_emb_matrix/sub_2Sub3Adam/update_user_embedding/u_org_emb_matrix/sub_2/x
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_org_emb_matrix/mul_1Mul>Adam/update_user_embedding/u_org_emb_matrix/UnsortedSegmentSum1Adam/update_user_embedding/u_org_emb_matrix/sub_2*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_org_emb_matrix/mul_2Mul)user_embedding/u_org_emb_matrix/Adam/read
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_org_emb_matrix/AssignAssign$user_embedding/u_org_emb_matrix/Adam1Adam/update_user_embedding/u_org_emb_matrix/mul_2*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
6Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd
ScatterAdd$user_embedding/u_org_emb_matrix/Adam2Adam/update_user_embedding/u_org_emb_matrix/Unique1Adam/update_user_embedding/u_org_emb_matrix/mul_13^Adam/update_user_embedding/u_org_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_org_emb_matrix/mul_3Mul>Adam/update_user_embedding/u_org_emb_matrix/UnsortedSegmentSum>Adam/update_user_embedding/u_org_emb_matrix/UnsortedSegmentSum*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*'
_output_shapes
:?????????

?
3Adam/update_user_embedding/u_org_emb_matrix/sub_3/xConst*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_org_emb_matrix/sub_3Sub3Adam/update_user_embedding/u_org_emb_matrix/sub_3/x
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_org_emb_matrix/mul_4Mul1Adam/update_user_embedding/u_org_emb_matrix/mul_31Adam/update_user_embedding/u_org_emb_matrix/sub_3*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_org_emb_matrix/mul_5Mul+user_embedding/u_org_emb_matrix/Adam_1/read
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
4Adam/update_user_embedding/u_org_emb_matrix/Assign_1Assign&user_embedding/u_org_emb_matrix/Adam_11Adam/update_user_embedding/u_org_emb_matrix/mul_5*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
8Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd_1
ScatterAdd&user_embedding/u_org_emb_matrix/Adam_12Adam/update_user_embedding/u_org_emb_matrix/Unique1Adam/update_user_embedding/u_org_emb_matrix/mul_45^Adam/update_user_embedding/u_org_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_org_emb_matrix/Sqrt_1Sqrt8Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd_1*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_org_emb_matrix/mul_6Mul3Adam/update_user_embedding/u_org_emb_matrix/truediv6Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
/Adam/update_user_embedding/u_org_emb_matrix/addAdd2Adam/update_user_embedding/u_org_emb_matrix/Sqrt_1Adam/epsilon*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_org_emb_matrix/truediv_1RealDiv1Adam/update_user_embedding/u_org_emb_matrix/mul_6/Adam/update_user_embedding/u_org_emb_matrix/add*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_org_emb_matrix/AssignSub	AssignSubuser_embedding/u_org_emb_matrix5Adam/update_user_embedding/u_org_emb_matrix/truediv_1*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_org_emb_matrix/group_depsNoOp6^Adam/update_user_embedding/u_org_emb_matrix/AssignSub7^Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd9^Adam/update_user_embedding/u_org_emb_matrix/ScatterAdd_1*2
_class(
&$loc:@user_embedding/u_org_emb_matrix
?
3Adam/update_user_embedding/u_seat_emb_matrix/UniqueUnique8gradients/user_embedding/u_seat_emb_layer_grad/Reshape_1*
out_idx0*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*2
_output_shapes 
:?????????:?????????
?
2Adam/update_user_embedding/u_seat_emb_matrix/ShapeShape3Adam/update_user_embedding/u_seat_emb_matrix/Unique*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
out_type0*
_output_shapes
:
?
@Adam/update_user_embedding/u_seat_emb_matrix/strided_slice/stackConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
BAdam/update_user_embedding/u_seat_emb_matrix/strided_slice/stack_1Const*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
BAdam/update_user_embedding/u_seat_emb_matrix/strided_slice/stack_2Const*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
:Adam/update_user_embedding/u_seat_emb_matrix/strided_sliceStridedSlice2Adam/update_user_embedding/u_seat_emb_matrix/Shape@Adam/update_user_embedding/u_seat_emb_matrix/strided_slice/stackBAdam/update_user_embedding/u_seat_emb_matrix/strided_slice/stack_1BAdam/update_user_embedding/u_seat_emb_matrix/strided_slice/stack_2*
T0*
Index0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
?Adam/update_user_embedding/u_seat_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum6gradients/user_embedding/u_seat_emb_layer_grad/Reshape5Adam/update_user_embedding/u_seat_emb_matrix/Unique:1:Adam/update_user_embedding/u_seat_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_seat_emb_matrix/sub/xConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
0Adam/update_user_embedding/u_seat_emb_matrix/subSub2Adam/update_user_embedding/u_seat_emb_matrix/sub/xbeta2_power/read*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_seat_emb_matrix/SqrtSqrt0Adam/update_user_embedding/u_seat_emb_matrix/sub*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_seat_emb_matrix/mulMulAdam/learning_rate1Adam/update_user_embedding/u_seat_emb_matrix/Sqrt*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_seat_emb_matrix/sub_1/xConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_seat_emb_matrix/sub_1Sub4Adam/update_user_embedding/u_seat_emb_matrix/sub_1/xbeta1_power/read*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_seat_emb_matrix/truedivRealDiv0Adam/update_user_embedding/u_seat_emb_matrix/mul2Adam/update_user_embedding/u_seat_emb_matrix/sub_1*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
4Adam/update_user_embedding/u_seat_emb_matrix/sub_2/xConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_seat_emb_matrix/sub_2Sub4Adam/update_user_embedding/u_seat_emb_matrix/sub_2/x
Adam/beta1*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_1Mul?Adam/update_user_embedding/u_seat_emb_matrix/UnsortedSegmentSum2Adam/update_user_embedding/u_seat_emb_matrix/sub_2*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_2Mul*user_embedding/u_seat_emb_matrix/Adam/read
Adam/beta1*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
3Adam/update_user_embedding/u_seat_emb_matrix/AssignAssign%user_embedding/u_seat_emb_matrix/Adam2Adam/update_user_embedding/u_seat_emb_matrix/mul_2*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
7Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd
ScatterAdd%user_embedding/u_seat_emb_matrix/Adam3Adam/update_user_embedding/u_seat_emb_matrix/Unique2Adam/update_user_embedding/u_seat_emb_matrix/mul_14^Adam/update_user_embedding/u_seat_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_3Mul?Adam/update_user_embedding/u_seat_emb_matrix/UnsortedSegmentSum?Adam/update_user_embedding/u_seat_emb_matrix/UnsortedSegmentSum*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*'
_output_shapes
:?????????

?
4Adam/update_user_embedding/u_seat_emb_matrix/sub_3/xConst*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
2Adam/update_user_embedding/u_seat_emb_matrix/sub_3Sub4Adam/update_user_embedding/u_seat_emb_matrix/sub_3/x
Adam/beta2*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes
: 
?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_4Mul2Adam/update_user_embedding/u_seat_emb_matrix/mul_32Adam/update_user_embedding/u_seat_emb_matrix/sub_3*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*'
_output_shapes
:?????????

?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_5Mul,user_embedding/u_seat_emb_matrix/Adam_1/read
Adam/beta2*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_seat_emb_matrix/Assign_1Assign'user_embedding/u_seat_emb_matrix/Adam_12Adam/update_user_embedding/u_seat_emb_matrix/mul_5*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
9Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd_1
ScatterAdd'user_embedding/u_seat_emb_matrix/Adam_13Adam/update_user_embedding/u_seat_emb_matrix/Unique2Adam/update_user_embedding/u_seat_emb_matrix/mul_46^Adam/update_user_embedding/u_seat_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
3Adam/update_user_embedding/u_seat_emb_matrix/Sqrt_1Sqrt9Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd_1*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_seat_emb_matrix/mul_6Mul4Adam/update_user_embedding/u_seat_emb_matrix/truediv7Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
0Adam/update_user_embedding/u_seat_emb_matrix/addAdd3Adam/update_user_embedding/u_seat_emb_matrix/Sqrt_1Adam/epsilon*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_seat_emb_matrix/truediv_1RealDiv2Adam/update_user_embedding/u_seat_emb_matrix/mul_60Adam/update_user_embedding/u_seat_emb_matrix/add*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_seat_emb_matrix/AssignSub	AssignSub user_embedding/u_seat_emb_matrix6Adam/update_user_embedding/u_seat_emb_matrix/truediv_1*
use_locking( *
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
_output_shapes

:

?
7Adam/update_user_embedding/u_seat_emb_matrix/group_depsNoOp7^Adam/update_user_embedding/u_seat_emb_matrix/AssignSub8^Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd:^Adam/update_user_embedding/u_seat_emb_matrix/ScatterAdd_1*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix
?
2Adam/update_user_embedding/u_pos_emb_matrix/UniqueUnique7gradients/user_embedding/u_pos_emb_layer_grad/Reshape_1*
out_idx0*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*2
_output_shapes 
:?????????:?????????
?
1Adam/update_user_embedding/u_pos_emb_matrix/ShapeShape2Adam/update_user_embedding/u_pos_emb_matrix/Unique*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
out_type0*
_output_shapes
:
?
?Adam/update_user_embedding/u_pos_emb_matrix/strided_slice/stackConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_pos_emb_matrix/strided_slice/stack_1Const*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
AAdam/update_user_embedding/u_pos_emb_matrix/strided_slice/stack_2Const*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
9Adam/update_user_embedding/u_pos_emb_matrix/strided_sliceStridedSlice1Adam/update_user_embedding/u_pos_emb_matrix/Shape?Adam/update_user_embedding/u_pos_emb_matrix/strided_slice/stackAAdam/update_user_embedding/u_pos_emb_matrix/strided_slice/stack_1AAdam/update_user_embedding/u_pos_emb_matrix/strided_slice/stack_2*
T0*
Index0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
>Adam/update_user_embedding/u_pos_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum5gradients/user_embedding/u_pos_emb_layer_grad/Reshape4Adam/update_user_embedding/u_pos_emb_matrix/Unique:19Adam/update_user_embedding/u_pos_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_pos_emb_matrix/sub/xConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
/Adam/update_user_embedding/u_pos_emb_matrix/subSub1Adam/update_user_embedding/u_pos_emb_matrix/sub/xbeta2_power/read*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
0Adam/update_user_embedding/u_pos_emb_matrix/SqrtSqrt/Adam/update_user_embedding/u_pos_emb_matrix/sub*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
/Adam/update_user_embedding/u_pos_emb_matrix/mulMulAdam/learning_rate0Adam/update_user_embedding/u_pos_emb_matrix/Sqrt*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_pos_emb_matrix/sub_1/xConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_pos_emb_matrix/sub_1Sub3Adam/update_user_embedding/u_pos_emb_matrix/sub_1/xbeta1_power/read*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_pos_emb_matrix/truedivRealDiv/Adam/update_user_embedding/u_pos_emb_matrix/mul1Adam/update_user_embedding/u_pos_emb_matrix/sub_1*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
3Adam/update_user_embedding/u_pos_emb_matrix/sub_2/xConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_pos_emb_matrix/sub_2Sub3Adam/update_user_embedding/u_pos_emb_matrix/sub_2/x
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_1Mul>Adam/update_user_embedding/u_pos_emb_matrix/UnsortedSegmentSum1Adam/update_user_embedding/u_pos_emb_matrix/sub_2*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_2Mul)user_embedding/u_pos_emb_matrix/Adam/read
Adam/beta1*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_pos_emb_matrix/AssignAssign$user_embedding/u_pos_emb_matrix/Adam1Adam/update_user_embedding/u_pos_emb_matrix/mul_2*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
6Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd
ScatterAdd$user_embedding/u_pos_emb_matrix/Adam2Adam/update_user_embedding/u_pos_emb_matrix/Unique1Adam/update_user_embedding/u_pos_emb_matrix/mul_13^Adam/update_user_embedding/u_pos_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_3Mul>Adam/update_user_embedding/u_pos_emb_matrix/UnsortedSegmentSum>Adam/update_user_embedding/u_pos_emb_matrix/UnsortedSegmentSum*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*'
_output_shapes
:?????????

?
3Adam/update_user_embedding/u_pos_emb_matrix/sub_3/xConst*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
1Adam/update_user_embedding/u_pos_emb_matrix/sub_3Sub3Adam/update_user_embedding/u_pos_emb_matrix/sub_3/x
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes
: 
?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_4Mul1Adam/update_user_embedding/u_pos_emb_matrix/mul_31Adam/update_user_embedding/u_pos_emb_matrix/sub_3*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*'
_output_shapes
:?????????

?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_5Mul+user_embedding/u_pos_emb_matrix/Adam_1/read
Adam/beta2*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
4Adam/update_user_embedding/u_pos_emb_matrix/Assign_1Assign&user_embedding/u_pos_emb_matrix/Adam_11Adam/update_user_embedding/u_pos_emb_matrix/mul_5*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
8Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd_1
ScatterAdd&user_embedding/u_pos_emb_matrix/Adam_12Adam/update_user_embedding/u_pos_emb_matrix/Unique1Adam/update_user_embedding/u_pos_emb_matrix/mul_45^Adam/update_user_embedding/u_pos_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
2Adam/update_user_embedding/u_pos_emb_matrix/Sqrt_1Sqrt8Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd_1*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
1Adam/update_user_embedding/u_pos_emb_matrix/mul_6Mul3Adam/update_user_embedding/u_pos_emb_matrix/truediv6Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
/Adam/update_user_embedding/u_pos_emb_matrix/addAdd2Adam/update_user_embedding/u_pos_emb_matrix/Sqrt_1Adam/epsilon*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_pos_emb_matrix/truediv_1RealDiv1Adam/update_user_embedding/u_pos_emb_matrix/mul_6/Adam/update_user_embedding/u_pos_emb_matrix/add*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
5Adam/update_user_embedding/u_pos_emb_matrix/AssignSub	AssignSubuser_embedding/u_pos_emb_matrix5Adam/update_user_embedding/u_pos_emb_matrix/truediv_1*
use_locking( *
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
_output_shapes

:

?
6Adam/update_user_embedding/u_pos_emb_matrix/group_depsNoOp6^Adam/update_user_embedding/u_pos_emb_matrix/AssignSub7^Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd9^Adam/update_user_embedding/u_pos_emb_matrix/ScatterAdd_1*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix
?
&Adam/update_u_type_fc/kernel/ApplyAdam	ApplyAdamu_type_fc/kernelu_type_fc/kernel/Adamu_type_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/user_fc/u_type_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*#
_class
loc:@u_type_fc/kernel*
use_nesterov( *
_output_shapes

:


?
$Adam/update_u_type_fc/bias/ApplyAdam	ApplyAdamu_type_fc/biasu_type_fc/bias/Adamu_type_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/user_fc/u_type_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@u_type_fc/bias*
use_nesterov( *
_output_shapes
:

?
%Adam/update_u_age_fc/kernel/ApplyAdam	ApplyAdamu_age_fc/kernelu_age_fc/kernel/Adamu_age_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/user_fc/u_age_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*"
_class
loc:@u_age_fc/kernel*
use_nesterov( *
_output_shapes

:


?
#Adam/update_u_age_fc/bias/ApplyAdam	ApplyAdamu_age_fc/biasu_age_fc/bias/Adamu_age_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/user_fc/u_age_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@u_age_fc/bias*
use_nesterov( *
_output_shapes
:

?
%Adam/update_u_sex_fc/kernel/ApplyAdam	ApplyAdamu_sex_fc/kernelu_sex_fc/kernel/Adamu_sex_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/user_fc/u_sex_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*"
_class
loc:@u_sex_fc/kernel*
use_nesterov( *
_output_shapes

:


?
#Adam/update_u_sex_fc/bias/ApplyAdam	ApplyAdamu_sex_fc/biasu_sex_fc/bias/Adamu_sex_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/user_fc/u_sex_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@u_sex_fc/bias*
use_nesterov( *
_output_shapes
:

?
%Adam/update_u_org_fc/kernel/ApplyAdam	ApplyAdamu_org_fc/kernelu_org_fc/kernel/Adamu_org_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/user_fc/u_org_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*"
_class
loc:@u_org_fc/kernel*
use_nesterov( *
_output_shapes

:


?
#Adam/update_u_org_fc/bias/ApplyAdam	ApplyAdamu_org_fc/biasu_org_fc/bias/Adamu_org_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/user_fc/u_org_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@u_org_fc/bias*
use_nesterov( *
_output_shapes
:

?
&Adam/update_u_seat_fc/kernel/ApplyAdam	ApplyAdamu_seat_fc/kernelu_seat_fc/kernel/Adamu_seat_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/user_fc/u_seat_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*#
_class
loc:@u_seat_fc/kernel*
use_nesterov( *
_output_shapes

:


?
$Adam/update_u_seat_fc/bias/ApplyAdam	ApplyAdamu_seat_fc/biasu_seat_fc/bias/Adamu_seat_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/user_fc/u_seat_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@u_seat_fc/bias*
use_nesterov( *
_output_shapes
:

?
%Adam/update_u_pos_id/kernel/ApplyAdam	ApplyAdamu_pos_id/kernelu_pos_id/kernel/Adamu_pos_id/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/user_fc/u_pos_id/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*"
_class
loc:@u_pos_id/kernel*
use_nesterov( *
_output_shapes

:


?
#Adam/update_u_pos_id/bias/ApplyAdam	ApplyAdamu_pos_id/biasu_pos_id/bias/Adamu_pos_id/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/user_fc/u_pos_id/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@u_pos_id/bias*
use_nesterov( *
_output_shapes
:

?
*Adam/update_u_concated_fc/kernel/ApplyAdam	ApplyAdamu_concated_fc/kernelu_concated_fc/kernel/Adamu_concated_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/user_fc/u_concated_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*'
_class
loc:@u_concated_fc/kernel*
use_nesterov( *
_output_shapes

:<@
?
(Adam/update_u_concated_fc/bias/ApplyAdam	ApplyAdamu_concated_fc/biasu_concated_fc/bias/Adamu_concated_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/user_fc/u_concated_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@u_concated_fc/bias*
use_nesterov( *
_output_shapes
:@
?
:Adam/update_item_class_embedding/i_class_emb_matrix/UniqueUnique?gradients/item_class_embedding/i_class_emb_layer_grad/Reshape_1*
out_idx0*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*2
_output_shapes 
:?????????:?????????
?
9Adam/update_item_class_embedding/i_class_emb_matrix/ShapeShape:Adam/update_item_class_embedding/i_class_emb_matrix/Unique*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
out_type0*
_output_shapes
:
?
GAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stackConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB: *
dtype0*
_output_shapes
:
?
IAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stack_1Const*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
IAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stack_2Const*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB:*
dtype0*
_output_shapes
:
?
AAdam/update_item_class_embedding/i_class_emb_matrix/strided_sliceStridedSlice9Adam/update_item_class_embedding/i_class_emb_matrix/ShapeGAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stackIAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stack_1IAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice/stack_2*
T0*
Index0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
FAdam/update_item_class_embedding/i_class_emb_matrix/UnsortedSegmentSumUnsortedSegmentSum=gradients/item_class_embedding/i_class_emb_layer_grad/Reshape<Adam/update_item_class_embedding/i_class_emb_matrix/Unique:1AAdam/update_item_class_embedding/i_class_emb_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*'
_output_shapes
:?????????
?
9Adam/update_item_class_embedding/i_class_emb_matrix/sub/xConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7Adam/update_item_class_embedding/i_class_emb_matrix/subSub9Adam/update_item_class_embedding/i_class_emb_matrix/sub/xbeta2_power/read*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
8Adam/update_item_class_embedding/i_class_emb_matrix/SqrtSqrt7Adam/update_item_class_embedding/i_class_emb_matrix/sub*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
7Adam/update_item_class_embedding/i_class_emb_matrix/mulMulAdam/learning_rate8Adam/update_item_class_embedding/i_class_emb_matrix/Sqrt*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
;Adam/update_item_class_embedding/i_class_emb_matrix/sub_1/xConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
9Adam/update_item_class_embedding/i_class_emb_matrix/sub_1Sub;Adam/update_item_class_embedding/i_class_emb_matrix/sub_1/xbeta1_power/read*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
;Adam/update_item_class_embedding/i_class_emb_matrix/truedivRealDiv7Adam/update_item_class_embedding/i_class_emb_matrix/mul9Adam/update_item_class_embedding/i_class_emb_matrix/sub_1*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
;Adam/update_item_class_embedding/i_class_emb_matrix/sub_2/xConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
9Adam/update_item_class_embedding/i_class_emb_matrix/sub_2Sub;Adam/update_item_class_embedding/i_class_emb_matrix/sub_2/x
Adam/beta1*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_1MulFAdam/update_item_class_embedding/i_class_emb_matrix/UnsortedSegmentSum9Adam/update_item_class_embedding/i_class_emb_matrix/sub_2*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*'
_output_shapes
:?????????
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_2Mul1item_class_embedding/i_class_emb_matrix/Adam/read
Adam/beta1*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
:Adam/update_item_class_embedding/i_class_emb_matrix/AssignAssign,item_class_embedding/i_class_emb_matrix/Adam9Adam/update_item_class_embedding/i_class_emb_matrix/mul_2*
use_locking( *
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
>Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAdd
ScatterAdd,item_class_embedding/i_class_emb_matrix/Adam:Adam/update_item_class_embedding/i_class_emb_matrix/Unique9Adam/update_item_class_embedding/i_class_emb_matrix/mul_1;^Adam/update_item_class_embedding/i_class_emb_matrix/Assign*
use_locking( *
Tindices0*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_3MulFAdam/update_item_class_embedding/i_class_emb_matrix/UnsortedSegmentSumFAdam/update_item_class_embedding/i_class_emb_matrix/UnsortedSegmentSum*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*'
_output_shapes
:?????????
?
;Adam/update_item_class_embedding/i_class_emb_matrix/sub_3/xConst*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
9Adam/update_item_class_embedding/i_class_emb_matrix/sub_3Sub;Adam/update_item_class_embedding/i_class_emb_matrix/sub_3/x
Adam/beta2*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes
: 
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_4Mul9Adam/update_item_class_embedding/i_class_emb_matrix/mul_39Adam/update_item_class_embedding/i_class_emb_matrix/sub_3*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*'
_output_shapes
:?????????
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_5Mul3item_class_embedding/i_class_emb_matrix/Adam_1/read
Adam/beta2*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
<Adam/update_item_class_embedding/i_class_emb_matrix/Assign_1Assign.item_class_embedding/i_class_emb_matrix/Adam_19Adam/update_item_class_embedding/i_class_emb_matrix/mul_5*
use_locking( *
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
@Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAdd_1
ScatterAdd.item_class_embedding/i_class_emb_matrix/Adam_1:Adam/update_item_class_embedding/i_class_emb_matrix/Unique9Adam/update_item_class_embedding/i_class_emb_matrix/mul_4=^Adam/update_item_class_embedding/i_class_emb_matrix/Assign_1*
use_locking( *
Tindices0*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
:Adam/update_item_class_embedding/i_class_emb_matrix/Sqrt_1Sqrt@Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAdd_1*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
9Adam/update_item_class_embedding/i_class_emb_matrix/mul_6Mul;Adam/update_item_class_embedding/i_class_emb_matrix/truediv>Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAdd*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
7Adam/update_item_class_embedding/i_class_emb_matrix/addAdd:Adam/update_item_class_embedding/i_class_emb_matrix/Sqrt_1Adam/epsilon*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
=Adam/update_item_class_embedding/i_class_emb_matrix/truediv_1RealDiv9Adam/update_item_class_embedding/i_class_emb_matrix/mul_67Adam/update_item_class_embedding/i_class_emb_matrix/add*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
=Adam/update_item_class_embedding/i_class_emb_matrix/AssignSub	AssignSub'item_class_embedding/i_class_emb_matrix=Adam/update_item_class_embedding/i_class_emb_matrix/truediv_1*
use_locking( *
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
_output_shapes

:
?
>Adam/update_item_class_embedding/i_class_emb_matrix/group_depsNoOp>^Adam/update_item_class_embedding/i_class_emb_matrix/AssignSub?^Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAddA^Adam/update_item_class_embedding/i_class_emb_matrix/ScatterAdd_1*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix
?
'Adam/update_i_class_fc/kernel/ApplyAdam	ApplyAdami_class_fc/kerneli_class_fc/kernel/Adami_class_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/item_class_fc/i_class_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*$
_class
loc:@i_class_fc/kernel*
use_nesterov( *
_output_shapes

:
?
%Adam/update_i_class_fc/bias/ApplyAdam	ApplyAdami_class_fc/biasi_class_fc/bias/Adami_class_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonJgradients/item_class_fc/i_class_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@i_class_fc/bias*
use_nesterov( *
_output_shapes
:
?
*Adam/update_i_concated_fc/kernel/ApplyAdam	ApplyAdami_concated_fc/kerneli_concated_fc/kernel/Adami_concated_fc/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMgradients/item_concated_fc/i_concated_fc/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*'
_class
loc:@i_concated_fc/kernel*
use_nesterov( *
_output_shapes

:@
?
(Adam/update_i_concated_fc/bias/ApplyAdam	ApplyAdami_concated_fc/biasi_concated_fc/bias/Adami_concated_fc/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonPgradients/item_concated_fc/i_concated_fc/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@i_concated_fc/bias*
use_nesterov( *
_output_shapes
:@
?
#Adam/update_expert_weight/ApplyAdam	ApplyAdamexpert_weightexpert_weight/Adamexpert_weight/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon5gradients/expert/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0* 
_class
loc:@expert_weight*
use_nesterov( *#
_output_shapes
:?
?
!Adam/update_expert_bias/ApplyAdam	ApplyAdamexpert_biasexpert_bias/Adamexpert_bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon4gradients/expert/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@expert_bias*
use_nesterov( *
_output_shapes
:
?
"Adam/update_gate1_weight/ApplyAdam	ApplyAdamgate1_weightgate1_weight/Adamgate1_weight/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/gate1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@gate1_weight*
use_nesterov( *
_output_shapes
:	?
?
 Adam/update_gate1_bias/ApplyAdam	ApplyAdam
gate1_biasgate1_bias/Adamgate1_bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/gate1/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@gate1_bias*
use_nesterov( *
_output_shapes
:
?
"Adam/update_gate2_weight/ApplyAdam	ApplyAdamgate2_weightgate2_weight/Adamgate2_weight/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/gate2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@gate2_weight*
use_nesterov( *
_output_shapes
:	?
?
 Adam/update_gate2_bias/ApplyAdam	ApplyAdam
gate2_biasgate2_bias/Adamgate2_bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/gate2/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@gate2_bias*
use_nesterov( *
_output_shapes
:
?
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/label1_output/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( *
_output_shapes

:
?
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/label1_output/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/bias*
use_nesterov( *
_output_shapes
:
?
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/label1_output/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( *
_output_shapes

: 
?
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/label1_output/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
use_nesterov( *
_output_shapes
: 
?
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/label1_output/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
use_nesterov( *
_output_shapes

: 
?
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/label1_output/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_2/bias*
use_nesterov( *
_output_shapes
:
?
$Adam/update_dense_3/kernel/ApplyAdam	ApplyAdamdense_3/kerneldense_3/kernel/Adamdense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/label2_output/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_3/kernel*
use_nesterov( *
_output_shapes

:
?
"Adam/update_dense_3/bias/ApplyAdam	ApplyAdamdense_3/biasdense_3/bias/Adamdense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/label2_output/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_3/bias*
use_nesterov( *
_output_shapes
:
?
$Adam/update_dense_5/kernel/ApplyAdam	ApplyAdamdense_5/kerneldense_5/kernel/Adamdense_5/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/label2_output/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_5/kernel*
use_nesterov( *
_output_shapes

:
?
"Adam/update_dense_5/bias/ApplyAdam	ApplyAdamdense_5/biasdense_5/bias/Adamdense_5/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/label2_output/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_5/bias*
use_nesterov( *
_output_shapes
:
?
Adam/mulMulbeta1_power/read
Adam/beta1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_3/bias/ApplyAdam%^Adam/update_dense_3/kernel/ApplyAdam#^Adam/update_dense_5/bias/ApplyAdam%^Adam/update_dense_5/kernel/ApplyAdam"^Adam/update_expert_bias/ApplyAdam$^Adam/update_expert_weight/ApplyAdam!^Adam/update_gate1_bias/ApplyAdam#^Adam/update_gate1_weight/ApplyAdam!^Adam/update_gate2_bias/ApplyAdam#^Adam/update_gate2_weight/ApplyAdam&^Adam/update_i_class_fc/bias/ApplyAdam(^Adam/update_i_class_fc/kernel/ApplyAdam)^Adam/update_i_concated_fc/bias/ApplyAdam+^Adam/update_i_concated_fc/kernel/ApplyAdam?^Adam/update_item_class_embedding/i_class_emb_matrix/group_deps$^Adam/update_u_age_fc/bias/ApplyAdam&^Adam/update_u_age_fc/kernel/ApplyAdam)^Adam/update_u_concated_fc/bias/ApplyAdam+^Adam/update_u_concated_fc/kernel/ApplyAdam$^Adam/update_u_org_fc/bias/ApplyAdam&^Adam/update_u_org_fc/kernel/ApplyAdam$^Adam/update_u_pos_id/bias/ApplyAdam&^Adam/update_u_pos_id/kernel/ApplyAdam%^Adam/update_u_seat_fc/bias/ApplyAdam'^Adam/update_u_seat_fc/kernel/ApplyAdam$^Adam/update_u_sex_fc/bias/ApplyAdam&^Adam/update_u_sex_fc/kernel/ApplyAdam%^Adam/update_u_type_fc/bias/ApplyAdam'^Adam/update_u_type_fc/kernel/ApplyAdam7^Adam/update_user_embedding/u_age_emn_matrix/group_deps7^Adam/update_user_embedding/u_org_emb_matrix/group_deps7^Adam/update_user_embedding/u_pos_emb_matrix/group_deps8^Adam/update_user_embedding/u_seat_emb_matrix/group_deps7^Adam/update_user_embedding/u_sex_emb_matrix/group_deps8^Adam/update_user_embedding/u_type_emb_matrix/group_deps*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_3/bias/ApplyAdam%^Adam/update_dense_3/kernel/ApplyAdam#^Adam/update_dense_5/bias/ApplyAdam%^Adam/update_dense_5/kernel/ApplyAdam"^Adam/update_expert_bias/ApplyAdam$^Adam/update_expert_weight/ApplyAdam!^Adam/update_gate1_bias/ApplyAdam#^Adam/update_gate1_weight/ApplyAdam!^Adam/update_gate2_bias/ApplyAdam#^Adam/update_gate2_weight/ApplyAdam&^Adam/update_i_class_fc/bias/ApplyAdam(^Adam/update_i_class_fc/kernel/ApplyAdam)^Adam/update_i_concated_fc/bias/ApplyAdam+^Adam/update_i_concated_fc/kernel/ApplyAdam?^Adam/update_item_class_embedding/i_class_emb_matrix/group_deps$^Adam/update_u_age_fc/bias/ApplyAdam&^Adam/update_u_age_fc/kernel/ApplyAdam)^Adam/update_u_concated_fc/bias/ApplyAdam+^Adam/update_u_concated_fc/kernel/ApplyAdam$^Adam/update_u_org_fc/bias/ApplyAdam&^Adam/update_u_org_fc/kernel/ApplyAdam$^Adam/update_u_pos_id/bias/ApplyAdam&^Adam/update_u_pos_id/kernel/ApplyAdam%^Adam/update_u_seat_fc/bias/ApplyAdam'^Adam/update_u_seat_fc/kernel/ApplyAdam$^Adam/update_u_sex_fc/bias/ApplyAdam&^Adam/update_u_sex_fc/kernel/ApplyAdam%^Adam/update_u_type_fc/bias/ApplyAdam'^Adam/update_u_type_fc/kernel/ApplyAdam7^Adam/update_user_embedding/u_age_emn_matrix/group_deps7^Adam/update_user_embedding/u_org_emb_matrix/group_deps7^Adam/update_user_embedding/u_pos_emb_matrix/group_deps8^Adam/update_user_embedding/u_seat_emb_matrix/group_deps7^Adam/update_user_embedding/u_sex_emb_matrix/group_deps8^Adam/update_user_embedding/u_type_emb_matrix/group_deps*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_3/bias/ApplyAdam%^Adam/update_dense_3/kernel/ApplyAdam#^Adam/update_dense_5/bias/ApplyAdam%^Adam/update_dense_5/kernel/ApplyAdam"^Adam/update_expert_bias/ApplyAdam$^Adam/update_expert_weight/ApplyAdam!^Adam/update_gate1_bias/ApplyAdam#^Adam/update_gate1_weight/ApplyAdam!^Adam/update_gate2_bias/ApplyAdam#^Adam/update_gate2_weight/ApplyAdam&^Adam/update_i_class_fc/bias/ApplyAdam(^Adam/update_i_class_fc/kernel/ApplyAdam)^Adam/update_i_concated_fc/bias/ApplyAdam+^Adam/update_i_concated_fc/kernel/ApplyAdam?^Adam/update_item_class_embedding/i_class_emb_matrix/group_deps$^Adam/update_u_age_fc/bias/ApplyAdam&^Adam/update_u_age_fc/kernel/ApplyAdam)^Adam/update_u_concated_fc/bias/ApplyAdam+^Adam/update_u_concated_fc/kernel/ApplyAdam$^Adam/update_u_org_fc/bias/ApplyAdam&^Adam/update_u_org_fc/kernel/ApplyAdam$^Adam/update_u_pos_id/bias/ApplyAdam&^Adam/update_u_pos_id/kernel/ApplyAdam%^Adam/update_u_seat_fc/bias/ApplyAdam'^Adam/update_u_seat_fc/kernel/ApplyAdam$^Adam/update_u_sex_fc/bias/ApplyAdam&^Adam/update_u_sex_fc/kernel/ApplyAdam%^Adam/update_u_type_fc/bias/ApplyAdam'^Adam/update_u_type_fc/kernel/ApplyAdam7^Adam/update_user_embedding/u_age_emn_matrix/group_deps7^Adam/update_user_embedding/u_org_emb_matrix/group_deps7^Adam/update_user_embedding/u_pos_emb_matrix/group_deps8^Adam/update_user_embedding/u_seat_emb_matrix/group_deps7^Adam/update_user_embedding/u_sex_emb_matrix/group_deps8^Adam/update_user_embedding/u_type_emb_matrix/group_deps
?
initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign^dense_2/bias/Adam/Assign^dense_2/bias/Adam_1/Assign^dense_2/bias/Assign^dense_2/kernel/Adam/Assign^dense_2/kernel/Adam_1/Assign^dense_2/kernel/Assign^dense_3/bias/Adam/Assign^dense_3/bias/Adam_1/Assign^dense_3/bias/Assign^dense_3/kernel/Adam/Assign^dense_3/kernel/Adam_1/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Adam/Assign^dense_5/bias/Adam_1/Assign^dense_5/bias/Assign^dense_5/kernel/Adam/Assign^dense_5/kernel/Adam_1/Assign^dense_5/kernel/Assign^expert_bias/Adam/Assign^expert_bias/Adam_1/Assign^expert_bias/Assign^expert_weight/Adam/Assign^expert_weight/Adam_1/Assign^expert_weight/Assign^gate1_bias/Adam/Assign^gate1_bias/Adam_1/Assign^gate1_bias/Assign^gate1_weight/Adam/Assign^gate1_weight/Adam_1/Assign^gate1_weight/Assign^gate2_bias/Adam/Assign^gate2_bias/Adam_1/Assign^gate2_bias/Assign^gate2_weight/Adam/Assign^gate2_weight/Adam_1/Assign^gate2_weight/Assign^i_class_fc/bias/Adam/Assign^i_class_fc/bias/Adam_1/Assign^i_class_fc/bias/Assign^i_class_fc/kernel/Adam/Assign ^i_class_fc/kernel/Adam_1/Assign^i_class_fc/kernel/Assign^i_concated_fc/bias/Adam/Assign!^i_concated_fc/bias/Adam_1/Assign^i_concated_fc/bias/Assign!^i_concated_fc/kernel/Adam/Assign#^i_concated_fc/kernel/Adam_1/Assign^i_concated_fc/kernel/Assign^i_entities_emb_fc/bias/Assign ^i_entities_emb_fc/kernel/Assign4^item_class_embedding/i_class_emb_matrix/Adam/Assign6^item_class_embedding/i_class_emb_matrix/Adam_1/Assign/^item_class_embedding/i_class_emb_matrix/Assign^u_age_fc/bias/Adam/Assign^u_age_fc/bias/Adam_1/Assign^u_age_fc/bias/Assign^u_age_fc/kernel/Adam/Assign^u_age_fc/kernel/Adam_1/Assign^u_age_fc/kernel/Assign^u_concated_fc/bias/Adam/Assign!^u_concated_fc/bias/Adam_1/Assign^u_concated_fc/bias/Assign!^u_concated_fc/kernel/Adam/Assign#^u_concated_fc/kernel/Adam_1/Assign^u_concated_fc/kernel/Assign^u_org_fc/bias/Adam/Assign^u_org_fc/bias/Adam_1/Assign^u_org_fc/bias/Assign^u_org_fc/kernel/Adam/Assign^u_org_fc/kernel/Adam_1/Assign^u_org_fc/kernel/Assign^u_pos_id/bias/Adam/Assign^u_pos_id/bias/Adam_1/Assign^u_pos_id/bias/Assign^u_pos_id/kernel/Adam/Assign^u_pos_id/kernel/Adam_1/Assign^u_pos_id/kernel/Assign^u_seat_fc/bias/Adam/Assign^u_seat_fc/bias/Adam_1/Assign^u_seat_fc/bias/Assign^u_seat_fc/kernel/Adam/Assign^u_seat_fc/kernel/Adam_1/Assign^u_seat_fc/kernel/Assign^u_sex_fc/bias/Adam/Assign^u_sex_fc/bias/Adam_1/Assign^u_sex_fc/bias/Assign^u_sex_fc/kernel/Adam/Assign^u_sex_fc/kernel/Adam_1/Assign^u_sex_fc/kernel/Assign^u_type_fc/bias/Adam/Assign^u_type_fc/bias/Adam_1/Assign^u_type_fc/bias/Assign^u_type_fc/kernel/Adam/Assign^u_type_fc/kernel/Adam_1/Assign^u_type_fc/kernel/Assign,^user_embedding/u_age_emn_matrix/Adam/Assign.^user_embedding/u_age_emn_matrix/Adam_1/Assign'^user_embedding/u_age_emn_matrix/Assign,^user_embedding/u_org_emb_matrix/Adam/Assign.^user_embedding/u_org_emb_matrix/Adam_1/Assign'^user_embedding/u_org_emb_matrix/Assign,^user_embedding/u_pos_emb_matrix/Adam/Assign.^user_embedding/u_pos_emb_matrix/Adam_1/Assign'^user_embedding/u_pos_emb_matrix/Assign-^user_embedding/u_seat_emb_matrix/Adam/Assign/^user_embedding/u_seat_emb_matrix/Adam_1/Assign(^user_embedding/u_seat_emb_matrix/Assign,^user_embedding/u_sex_emb_matrix/Adam/Assign.^user_embedding/u_sex_emb_matrix/Adam_1/Assign'^user_embedding/u_sex_emb_matrix/Assign-^user_embedding/u_type_emb_matrix/Adam/Assign/^user_embedding/u_type_emb_matrix/Adam_1/Assign(^user_embedding/u_type_emb_matrix/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_dfee5e38c93b4e7db75badb142083553/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*?
value?B??Bbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1Bdense_3/biasBdense_3/bias/AdamBdense_3/bias/Adam_1Bdense_3/kernelBdense_3/kernel/AdamBdense_3/kernel/Adam_1Bdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/bias/AdamBdense_5/bias/Adam_1Bdense_5/kernelBdense_5/kernel/AdamBdense_5/kernel/Adam_1Bexpert_biasBexpert_bias/AdamBexpert_bias/Adam_1Bexpert_weightBexpert_weight/AdamBexpert_weight/Adam_1B
gate1_biasBgate1_bias/AdamBgate1_bias/Adam_1Bgate1_weightBgate1_weight/AdamBgate1_weight/Adam_1B
gate2_biasBgate2_bias/AdamBgate2_bias/Adam_1Bgate2_weightBgate2_weight/AdamBgate2_weight/Adam_1Bi_class_fc/biasBi_class_fc/bias/AdamBi_class_fc/bias/Adam_1Bi_class_fc/kernelBi_class_fc/kernel/AdamBi_class_fc/kernel/Adam_1Bi_concated_fc/biasBi_concated_fc/bias/AdamBi_concated_fc/bias/Adam_1Bi_concated_fc/kernelBi_concated_fc/kernel/AdamBi_concated_fc/kernel/Adam_1Bi_entities_emb_fc/biasBi_entities_emb_fc/kernelB'item_class_embedding/i_class_emb_matrixB,item_class_embedding/i_class_emb_matrix/AdamB.item_class_embedding/i_class_emb_matrix/Adam_1Bu_age_fc/biasBu_age_fc/bias/AdamBu_age_fc/bias/Adam_1Bu_age_fc/kernelBu_age_fc/kernel/AdamBu_age_fc/kernel/Adam_1Bu_concated_fc/biasBu_concated_fc/bias/AdamBu_concated_fc/bias/Adam_1Bu_concated_fc/kernelBu_concated_fc/kernel/AdamBu_concated_fc/kernel/Adam_1Bu_org_fc/biasBu_org_fc/bias/AdamBu_org_fc/bias/Adam_1Bu_org_fc/kernelBu_org_fc/kernel/AdamBu_org_fc/kernel/Adam_1Bu_pos_id/biasBu_pos_id/bias/AdamBu_pos_id/bias/Adam_1Bu_pos_id/kernelBu_pos_id/kernel/AdamBu_pos_id/kernel/Adam_1Bu_seat_fc/biasBu_seat_fc/bias/AdamBu_seat_fc/bias/Adam_1Bu_seat_fc/kernelBu_seat_fc/kernel/AdamBu_seat_fc/kernel/Adam_1Bu_sex_fc/biasBu_sex_fc/bias/AdamBu_sex_fc/bias/Adam_1Bu_sex_fc/kernelBu_sex_fc/kernel/AdamBu_sex_fc/kernel/Adam_1Bu_type_fc/biasBu_type_fc/bias/AdamBu_type_fc/bias/Adam_1Bu_type_fc/kernelBu_type_fc/kernel/AdamBu_type_fc/kernel/Adam_1Buser_embedding/u_age_emn_matrixB$user_embedding/u_age_emn_matrix/AdamB&user_embedding/u_age_emn_matrix/Adam_1Buser_embedding/u_org_emb_matrixB$user_embedding/u_org_emb_matrix/AdamB&user_embedding/u_org_emb_matrix/Adam_1Buser_embedding/u_pos_emb_matrixB$user_embedding/u_pos_emb_matrix/AdamB&user_embedding/u_pos_emb_matrix/Adam_1B user_embedding/u_seat_emb_matrixB%user_embedding/u_seat_emb_matrix/AdamB'user_embedding/u_seat_emb_matrix/Adam_1Buser_embedding/u_sex_emb_matrixB$user_embedding/u_sex_emb_matrix/AdamB&user_embedding/u_sex_emb_matrix/Adam_1B user_embedding/u_type_emb_matrixB%user_embedding/u_type_emb_matrix/AdamB'user_embedding/u_type_emb_matrix/Adam_1*
dtype0*
_output_shapes	
:?
?
save/SaveV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1dense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1dense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1dense_3/biasdense_3/bias/Adamdense_3/bias/Adam_1dense_3/kerneldense_3/kernel/Adamdense_3/kernel/Adam_1dense_4/biasdense_4/kerneldense_5/biasdense_5/bias/Adamdense_5/bias/Adam_1dense_5/kerneldense_5/kernel/Adamdense_5/kernel/Adam_1expert_biasexpert_bias/Adamexpert_bias/Adam_1expert_weightexpert_weight/Adamexpert_weight/Adam_1
gate1_biasgate1_bias/Adamgate1_bias/Adam_1gate1_weightgate1_weight/Adamgate1_weight/Adam_1
gate2_biasgate2_bias/Adamgate2_bias/Adam_1gate2_weightgate2_weight/Adamgate2_weight/Adam_1i_class_fc/biasi_class_fc/bias/Adami_class_fc/bias/Adam_1i_class_fc/kerneli_class_fc/kernel/Adami_class_fc/kernel/Adam_1i_concated_fc/biasi_concated_fc/bias/Adami_concated_fc/bias/Adam_1i_concated_fc/kerneli_concated_fc/kernel/Adami_concated_fc/kernel/Adam_1i_entities_emb_fc/biasi_entities_emb_fc/kernel'item_class_embedding/i_class_emb_matrix,item_class_embedding/i_class_emb_matrix/Adam.item_class_embedding/i_class_emb_matrix/Adam_1u_age_fc/biasu_age_fc/bias/Adamu_age_fc/bias/Adam_1u_age_fc/kernelu_age_fc/kernel/Adamu_age_fc/kernel/Adam_1u_concated_fc/biasu_concated_fc/bias/Adamu_concated_fc/bias/Adam_1u_concated_fc/kernelu_concated_fc/kernel/Adamu_concated_fc/kernel/Adam_1u_org_fc/biasu_org_fc/bias/Adamu_org_fc/bias/Adam_1u_org_fc/kernelu_org_fc/kernel/Adamu_org_fc/kernel/Adam_1u_pos_id/biasu_pos_id/bias/Adamu_pos_id/bias/Adam_1u_pos_id/kernelu_pos_id/kernel/Adamu_pos_id/kernel/Adam_1u_seat_fc/biasu_seat_fc/bias/Adamu_seat_fc/bias/Adam_1u_seat_fc/kernelu_seat_fc/kernel/Adamu_seat_fc/kernel/Adam_1u_sex_fc/biasu_sex_fc/bias/Adamu_sex_fc/bias/Adam_1u_sex_fc/kernelu_sex_fc/kernel/Adamu_sex_fc/kernel/Adam_1u_type_fc/biasu_type_fc/bias/Adamu_type_fc/bias/Adam_1u_type_fc/kernelu_type_fc/kernel/Adamu_type_fc/kernel/Adam_1user_embedding/u_age_emn_matrix$user_embedding/u_age_emn_matrix/Adam&user_embedding/u_age_emn_matrix/Adam_1user_embedding/u_org_emb_matrix$user_embedding/u_org_emb_matrix/Adam&user_embedding/u_org_emb_matrix/Adam_1user_embedding/u_pos_emb_matrix$user_embedding/u_pos_emb_matrix/Adam&user_embedding/u_pos_emb_matrix/Adam_1 user_embedding/u_seat_emb_matrix%user_embedding/u_seat_emb_matrix/Adam'user_embedding/u_seat_emb_matrix/Adam_1user_embedding/u_sex_emb_matrix$user_embedding/u_sex_emb_matrix/Adam&user_embedding/u_sex_emb_matrix/Adam_1 user_embedding/u_type_emb_matrix%user_embedding/u_type_emb_matrix/Adam'user_embedding/u_type_emb_matrix/Adam_1*?
dtypes?
?2?
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst*?
value?B??Bbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1Bdense_3/biasBdense_3/bias/AdamBdense_3/bias/Adam_1Bdense_3/kernelBdense_3/kernel/AdamBdense_3/kernel/Adam_1Bdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/bias/AdamBdense_5/bias/Adam_1Bdense_5/kernelBdense_5/kernel/AdamBdense_5/kernel/Adam_1Bexpert_biasBexpert_bias/AdamBexpert_bias/Adam_1Bexpert_weightBexpert_weight/AdamBexpert_weight/Adam_1B
gate1_biasBgate1_bias/AdamBgate1_bias/Adam_1Bgate1_weightBgate1_weight/AdamBgate1_weight/Adam_1B
gate2_biasBgate2_bias/AdamBgate2_bias/Adam_1Bgate2_weightBgate2_weight/AdamBgate2_weight/Adam_1Bi_class_fc/biasBi_class_fc/bias/AdamBi_class_fc/bias/Adam_1Bi_class_fc/kernelBi_class_fc/kernel/AdamBi_class_fc/kernel/Adam_1Bi_concated_fc/biasBi_concated_fc/bias/AdamBi_concated_fc/bias/Adam_1Bi_concated_fc/kernelBi_concated_fc/kernel/AdamBi_concated_fc/kernel/Adam_1Bi_entities_emb_fc/biasBi_entities_emb_fc/kernelB'item_class_embedding/i_class_emb_matrixB,item_class_embedding/i_class_emb_matrix/AdamB.item_class_embedding/i_class_emb_matrix/Adam_1Bu_age_fc/biasBu_age_fc/bias/AdamBu_age_fc/bias/Adam_1Bu_age_fc/kernelBu_age_fc/kernel/AdamBu_age_fc/kernel/Adam_1Bu_concated_fc/biasBu_concated_fc/bias/AdamBu_concated_fc/bias/Adam_1Bu_concated_fc/kernelBu_concated_fc/kernel/AdamBu_concated_fc/kernel/Adam_1Bu_org_fc/biasBu_org_fc/bias/AdamBu_org_fc/bias/Adam_1Bu_org_fc/kernelBu_org_fc/kernel/AdamBu_org_fc/kernel/Adam_1Bu_pos_id/biasBu_pos_id/bias/AdamBu_pos_id/bias/Adam_1Bu_pos_id/kernelBu_pos_id/kernel/AdamBu_pos_id/kernel/Adam_1Bu_seat_fc/biasBu_seat_fc/bias/AdamBu_seat_fc/bias/Adam_1Bu_seat_fc/kernelBu_seat_fc/kernel/AdamBu_seat_fc/kernel/Adam_1Bu_sex_fc/biasBu_sex_fc/bias/AdamBu_sex_fc/bias/Adam_1Bu_sex_fc/kernelBu_sex_fc/kernel/AdamBu_sex_fc/kernel/Adam_1Bu_type_fc/biasBu_type_fc/bias/AdamBu_type_fc/bias/Adam_1Bu_type_fc/kernelBu_type_fc/kernel/AdamBu_type_fc/kernel/Adam_1Buser_embedding/u_age_emn_matrixB$user_embedding/u_age_emn_matrix/AdamB&user_embedding/u_age_emn_matrix/Adam_1Buser_embedding/u_org_emb_matrixB$user_embedding/u_org_emb_matrix/AdamB&user_embedding/u_org_emb_matrix/Adam_1Buser_embedding/u_pos_emb_matrixB$user_embedding/u_pos_emb_matrix/AdamB&user_embedding/u_pos_emb_matrix/Adam_1B user_embedding/u_seat_emb_matrixB%user_embedding/u_seat_emb_matrix/AdamB'user_embedding/u_seat_emb_matrix/Adam_1Buser_embedding/u_sex_emb_matrixB$user_embedding/u_sex_emb_matrix/AdamB&user_embedding/u_sex_emb_matrix/Adam_1B user_embedding/u_type_emb_matrixB%user_embedding/u_type_emb_matrix/AdamB'user_embedding/u_type_emb_matrix/Adam_1*
dtype0*
_output_shapes	
:?
?
save/RestoreV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
dtypes?
?2?*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_2Assign
dense/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_5Assigndense/kernelsave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_8Assigndense_1/biassave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_9Assigndense_1/bias/Adamsave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_10Assigndense_1/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_11Assigndense_1/kernelsave/RestoreV2:11*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_12Assigndense_1/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_13Assigndense_1/kernel/Adam_1save/RestoreV2:13*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_14Assigndense_2/biassave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_15Assigndense_2/bias/Adamsave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_16Assigndense_2/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_17Assigndense_2/kernelsave/RestoreV2:17*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_18Assigndense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_19Assigndense_2/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_20Assigndense_3/biassave/RestoreV2:20*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_21Assigndense_3/bias/Adamsave/RestoreV2:21*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_22Assigndense_3/bias/Adam_1save/RestoreV2:22*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_23Assigndense_3/kernelsave/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_24Assigndense_3/kernel/Adamsave/RestoreV2:24*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_25Assigndense_3/kernel/Adam_1save/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_26Assigndense_4/biassave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_27Assigndense_4/kernelsave/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

: 
?
save/Assign_28Assigndense_5/biassave/RestoreV2:28*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_29Assigndense_5/bias/Adamsave/RestoreV2:29*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_30Assigndense_5/bias/Adam_1save/RestoreV2:30*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_31Assigndense_5/kernelsave/RestoreV2:31*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_32Assigndense_5/kernel/Adamsave/RestoreV2:32*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_33Assigndense_5/kernel/Adam_1save/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_34Assignexpert_biassave/RestoreV2:34*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_35Assignexpert_bias/Adamsave/RestoreV2:35*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_36Assignexpert_bias/Adam_1save/RestoreV2:36*
use_locking(*
T0*
_class
loc:@expert_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_37Assignexpert_weightsave/RestoreV2:37*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
?
save/Assign_38Assignexpert_weight/Adamsave/RestoreV2:38*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
?
save/Assign_39Assignexpert_weight/Adam_1save/RestoreV2:39*
use_locking(*
T0* 
_class
loc:@expert_weight*
validate_shape(*#
_output_shapes
:?
?
save/Assign_40Assign
gate1_biassave/RestoreV2:40*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_41Assigngate1_bias/Adamsave/RestoreV2:41*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_42Assigngate1_bias/Adam_1save/RestoreV2:42*
use_locking(*
T0*
_class
loc:@gate1_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_43Assigngate1_weightsave/RestoreV2:43*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_44Assigngate1_weight/Adamsave/RestoreV2:44*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_45Assigngate1_weight/Adam_1save/RestoreV2:45*
use_locking(*
T0*
_class
loc:@gate1_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_46Assign
gate2_biassave/RestoreV2:46*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_47Assigngate2_bias/Adamsave/RestoreV2:47*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_48Assigngate2_bias/Adam_1save/RestoreV2:48*
use_locking(*
T0*
_class
loc:@gate2_bias*
validate_shape(*
_output_shapes
:
?
save/Assign_49Assigngate2_weightsave/RestoreV2:49*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_50Assigngate2_weight/Adamsave/RestoreV2:50*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_51Assigngate2_weight/Adam_1save/RestoreV2:51*
use_locking(*
T0*
_class
loc:@gate2_weight*
validate_shape(*
_output_shapes
:	?
?
save/Assign_52Assigni_class_fc/biassave/RestoreV2:52*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_53Assigni_class_fc/bias/Adamsave/RestoreV2:53*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_54Assigni_class_fc/bias/Adam_1save/RestoreV2:54*
use_locking(*
T0*"
_class
loc:@i_class_fc/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_55Assigni_class_fc/kernelsave/RestoreV2:55*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_56Assigni_class_fc/kernel/Adamsave/RestoreV2:56*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_57Assigni_class_fc/kernel/Adam_1save/RestoreV2:57*
use_locking(*
T0*$
_class
loc:@i_class_fc/kernel*
validate_shape(*
_output_shapes

:
?
save/Assign_58Assigni_concated_fc/biassave/RestoreV2:58*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_59Assigni_concated_fc/bias/Adamsave/RestoreV2:59*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_60Assigni_concated_fc/bias/Adam_1save/RestoreV2:60*
use_locking(*
T0*%
_class
loc:@i_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_61Assigni_concated_fc/kernelsave/RestoreV2:61*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
save/Assign_62Assigni_concated_fc/kernel/Adamsave/RestoreV2:62*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
save/Assign_63Assigni_concated_fc/kernel/Adam_1save/RestoreV2:63*
use_locking(*
T0*'
_class
loc:@i_concated_fc/kernel*
validate_shape(*
_output_shapes

:@
?
save/Assign_64Assigni_entities_emb_fc/biassave/RestoreV2:64*
use_locking(*
T0*)
_class
loc:@i_entities_emb_fc/bias*
validate_shape(*
_output_shapes
: 
?
save/Assign_65Assigni_entities_emb_fc/kernelsave/RestoreV2:65*
use_locking(*
T0*+
_class!
loc:@i_entities_emb_fc/kernel*
validate_shape(*
_output_shapes

:@ 
?
save/Assign_66Assign'item_class_embedding/i_class_emb_matrixsave/RestoreV2:66*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
save/Assign_67Assign,item_class_embedding/i_class_emb_matrix/Adamsave/RestoreV2:67*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
save/Assign_68Assign.item_class_embedding/i_class_emb_matrix/Adam_1save/RestoreV2:68*
use_locking(*
T0*:
_class0
.,loc:@item_class_embedding/i_class_emb_matrix*
validate_shape(*
_output_shapes

:
?
save/Assign_69Assignu_age_fc/biassave/RestoreV2:69*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_70Assignu_age_fc/bias/Adamsave/RestoreV2:70*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_71Assignu_age_fc/bias/Adam_1save/RestoreV2:71*
use_locking(*
T0* 
_class
loc:@u_age_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_72Assignu_age_fc/kernelsave/RestoreV2:72*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_73Assignu_age_fc/kernel/Adamsave/RestoreV2:73*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_74Assignu_age_fc/kernel/Adam_1save/RestoreV2:74*
use_locking(*
T0*"
_class
loc:@u_age_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_75Assignu_concated_fc/biassave/RestoreV2:75*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_76Assignu_concated_fc/bias/Adamsave/RestoreV2:76*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_77Assignu_concated_fc/bias/Adam_1save/RestoreV2:77*
use_locking(*
T0*%
_class
loc:@u_concated_fc/bias*
validate_shape(*
_output_shapes
:@
?
save/Assign_78Assignu_concated_fc/kernelsave/RestoreV2:78*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
save/Assign_79Assignu_concated_fc/kernel/Adamsave/RestoreV2:79*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
save/Assign_80Assignu_concated_fc/kernel/Adam_1save/RestoreV2:80*
use_locking(*
T0*'
_class
loc:@u_concated_fc/kernel*
validate_shape(*
_output_shapes

:<@
?
save/Assign_81Assignu_org_fc/biassave/RestoreV2:81*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_82Assignu_org_fc/bias/Adamsave/RestoreV2:82*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_83Assignu_org_fc/bias/Adam_1save/RestoreV2:83*
use_locking(*
T0* 
_class
loc:@u_org_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_84Assignu_org_fc/kernelsave/RestoreV2:84*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_85Assignu_org_fc/kernel/Adamsave/RestoreV2:85*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_86Assignu_org_fc/kernel/Adam_1save/RestoreV2:86*
use_locking(*
T0*"
_class
loc:@u_org_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_87Assignu_pos_id/biassave/RestoreV2:87*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_88Assignu_pos_id/bias/Adamsave/RestoreV2:88*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_89Assignu_pos_id/bias/Adam_1save/RestoreV2:89*
use_locking(*
T0* 
_class
loc:@u_pos_id/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_90Assignu_pos_id/kernelsave/RestoreV2:90*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_91Assignu_pos_id/kernel/Adamsave/RestoreV2:91*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_92Assignu_pos_id/kernel/Adam_1save/RestoreV2:92*
use_locking(*
T0*"
_class
loc:@u_pos_id/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_93Assignu_seat_fc/biassave/RestoreV2:93*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_94Assignu_seat_fc/bias/Adamsave/RestoreV2:94*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_95Assignu_seat_fc/bias/Adam_1save/RestoreV2:95*
use_locking(*
T0*!
_class
loc:@u_seat_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_96Assignu_seat_fc/kernelsave/RestoreV2:96*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_97Assignu_seat_fc/kernel/Adamsave/RestoreV2:97*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_98Assignu_seat_fc/kernel/Adam_1save/RestoreV2:98*
use_locking(*
T0*#
_class
loc:@u_seat_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_99Assignu_sex_fc/biassave/RestoreV2:99*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_100Assignu_sex_fc/bias/Adamsave/RestoreV2:100*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_101Assignu_sex_fc/bias/Adam_1save/RestoreV2:101*
use_locking(*
T0* 
_class
loc:@u_sex_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_102Assignu_sex_fc/kernelsave/RestoreV2:102*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_103Assignu_sex_fc/kernel/Adamsave/RestoreV2:103*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_104Assignu_sex_fc/kernel/Adam_1save/RestoreV2:104*
use_locking(*
T0*"
_class
loc:@u_sex_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_105Assignu_type_fc/biassave/RestoreV2:105*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_106Assignu_type_fc/bias/Adamsave/RestoreV2:106*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_107Assignu_type_fc/bias/Adam_1save/RestoreV2:107*
use_locking(*
T0*!
_class
loc:@u_type_fc/bias*
validate_shape(*
_output_shapes
:

?
save/Assign_108Assignu_type_fc/kernelsave/RestoreV2:108*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_109Assignu_type_fc/kernel/Adamsave/RestoreV2:109*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_110Assignu_type_fc/kernel/Adam_1save/RestoreV2:110*
use_locking(*
T0*#
_class
loc:@u_type_fc/kernel*
validate_shape(*
_output_shapes

:


?
save/Assign_111Assignuser_embedding/u_age_emn_matrixsave/RestoreV2:111*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_112Assign$user_embedding/u_age_emn_matrix/Adamsave/RestoreV2:112*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_113Assign&user_embedding/u_age_emn_matrix/Adam_1save/RestoreV2:113*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_age_emn_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_114Assignuser_embedding/u_org_emb_matrixsave/RestoreV2:114*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_115Assign$user_embedding/u_org_emb_matrix/Adamsave/RestoreV2:115*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_116Assign&user_embedding/u_org_emb_matrix/Adam_1save/RestoreV2:116*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_org_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_117Assignuser_embedding/u_pos_emb_matrixsave/RestoreV2:117*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_118Assign$user_embedding/u_pos_emb_matrix/Adamsave/RestoreV2:118*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_119Assign&user_embedding/u_pos_emb_matrix/Adam_1save/RestoreV2:119*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_pos_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_120Assign user_embedding/u_seat_emb_matrixsave/RestoreV2:120*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_121Assign%user_embedding/u_seat_emb_matrix/Adamsave/RestoreV2:121*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_122Assign'user_embedding/u_seat_emb_matrix/Adam_1save/RestoreV2:122*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_seat_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_123Assignuser_embedding/u_sex_emb_matrixsave/RestoreV2:123*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_124Assign$user_embedding/u_sex_emb_matrix/Adamsave/RestoreV2:124*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_125Assign&user_embedding/u_sex_emb_matrix/Adam_1save/RestoreV2:125*
use_locking(*
T0*2
_class(
&$loc:@user_embedding/u_sex_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_126Assign user_embedding/u_type_emb_matrixsave/RestoreV2:126*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_127Assign%user_embedding/u_type_emb_matrix/Adamsave/RestoreV2:127*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/Assign_128Assign'user_embedding/u_type_emb_matrix/Adam_1save/RestoreV2:128*
use_locking(*
T0*3
_class)
'%loc:@user_embedding/u_type_emb_matrix*
validate_shape(*
_output_shapes

:

?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8"??
	variables????
?
"user_embedding/u_type_emb_matrix:0'user_embedding/u_type_emb_matrix/Assign'user_embedding/u_type_emb_matrix/read:02user_embedding/random_uniform:08
?
!user_embedding/u_age_emn_matrix:0&user_embedding/u_age_emn_matrix/Assign&user_embedding/u_age_emn_matrix/read:02!user_embedding/random_uniform_1:08
?
!user_embedding/u_sex_emb_matrix:0&user_embedding/u_sex_emb_matrix/Assign&user_embedding/u_sex_emb_matrix/read:02!user_embedding/random_uniform_2:08
?
!user_embedding/u_org_emb_matrix:0&user_embedding/u_org_emb_matrix/Assign&user_embedding/u_org_emb_matrix/read:02!user_embedding/random_uniform_3:08
?
"user_embedding/u_seat_emb_matrix:0'user_embedding/u_seat_emb_matrix/Assign'user_embedding/u_seat_emb_matrix/read:02!user_embedding/random_uniform_4:08
?
!user_embedding/u_pos_emb_matrix:0&user_embedding/u_pos_emb_matrix/Assign&user_embedding/u_pos_emb_matrix/read:02!user_embedding/random_uniform_5:08
w
u_type_fc/kernel:0u_type_fc/kernel/Assignu_type_fc/kernel/read:02-u_type_fc/kernel/Initializer/random_uniform:08
f
u_type_fc/bias:0u_type_fc/bias/Assignu_type_fc/bias/read:02"u_type_fc/bias/Initializer/zeros:08
s
u_age_fc/kernel:0u_age_fc/kernel/Assignu_age_fc/kernel/read:02,u_age_fc/kernel/Initializer/random_uniform:08
b
u_age_fc/bias:0u_age_fc/bias/Assignu_age_fc/bias/read:02!u_age_fc/bias/Initializer/zeros:08
s
u_sex_fc/kernel:0u_sex_fc/kernel/Assignu_sex_fc/kernel/read:02,u_sex_fc/kernel/Initializer/random_uniform:08
b
u_sex_fc/bias:0u_sex_fc/bias/Assignu_sex_fc/bias/read:02!u_sex_fc/bias/Initializer/zeros:08
s
u_org_fc/kernel:0u_org_fc/kernel/Assignu_org_fc/kernel/read:02,u_org_fc/kernel/Initializer/random_uniform:08
b
u_org_fc/bias:0u_org_fc/bias/Assignu_org_fc/bias/read:02!u_org_fc/bias/Initializer/zeros:08
w
u_seat_fc/kernel:0u_seat_fc/kernel/Assignu_seat_fc/kernel/read:02-u_seat_fc/kernel/Initializer/random_uniform:08
f
u_seat_fc/bias:0u_seat_fc/bias/Assignu_seat_fc/bias/read:02"u_seat_fc/bias/Initializer/zeros:08
s
u_pos_id/kernel:0u_pos_id/kernel/Assignu_pos_id/kernel/read:02,u_pos_id/kernel/Initializer/random_uniform:08
b
u_pos_id/bias:0u_pos_id/bias/Assignu_pos_id/bias/read:02!u_pos_id/bias/Initializer/zeros:08
?
u_concated_fc/kernel:0u_concated_fc/kernel/Assignu_concated_fc/kernel/read:021u_concated_fc/kernel/Initializer/random_uniform:08
v
u_concated_fc/bias:0u_concated_fc/bias/Assignu_concated_fc/bias/read:02&u_concated_fc/bias/Initializer/zeros:08
?
)item_class_embedding/i_class_emb_matrix:0.item_class_embedding/i_class_emb_matrix/Assign.item_class_embedding/i_class_emb_matrix/read:02%item_class_embedding/random_uniform:08
{
i_class_fc/kernel:0i_class_fc/kernel/Assigni_class_fc/kernel/read:02.i_class_fc/kernel/Initializer/random_uniform:08
j
i_class_fc/bias:0i_class_fc/bias/Assigni_class_fc/bias/read:02#i_class_fc/bias/Initializer/zeros:08
?
i_entities_emb_fc/kernel:0i_entities_emb_fc/kernel/Assigni_entities_emb_fc/kernel/read:025i_entities_emb_fc/kernel/Initializer/random_uniform:08
?
i_entities_emb_fc/bias:0i_entities_emb_fc/bias/Assigni_entities_emb_fc/bias/read:02*i_entities_emb_fc/bias/Initializer/zeros:08
?
i_concated_fc/kernel:0i_concated_fc/kernel/Assigni_concated_fc/kernel/read:021i_concated_fc/kernel/Initializer/random_uniform:08
v
i_concated_fc/bias:0i_concated_fc/bias/Assigni_concated_fc/bias/read:02&i_concated_fc/bias/Initializer/zeros:08
m
expert_weight:0expert_weight/Assignexpert_weight/read:02,expert_weight/Initializer/truncated_normal:08
Z
expert_bias:0expert_bias/Assignexpert_bias/read:02expert_bias/Initializer/zeros:08
i
gate1_weight:0gate1_weight/Assigngate1_weight/read:02+gate1_weight/Initializer/truncated_normal:08
V
gate1_bias:0gate1_bias/Assigngate1_bias/read:02gate1_bias/Initializer/zeros:08
i
gate2_weight:0gate2_weight/Assigngate2_weight/read:02+gate2_weight/Initializer/truncated_normal:08
V
gate2_bias:0gate2_bias/Assigngate2_bias/read:02gate2_bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08
o
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:08
^
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:08
o
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:08
^
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
'user_embedding/u_type_emb_matrix/Adam:0,user_embedding/u_type_emb_matrix/Adam/Assign,user_embedding/u_type_emb_matrix/Adam/read:029user_embedding/u_type_emb_matrix/Adam/Initializer/zeros:0
?
)user_embedding/u_type_emb_matrix/Adam_1:0.user_embedding/u_type_emb_matrix/Adam_1/Assign.user_embedding/u_type_emb_matrix/Adam_1/read:02;user_embedding/u_type_emb_matrix/Adam_1/Initializer/zeros:0
?
&user_embedding/u_age_emn_matrix/Adam:0+user_embedding/u_age_emn_matrix/Adam/Assign+user_embedding/u_age_emn_matrix/Adam/read:028user_embedding/u_age_emn_matrix/Adam/Initializer/zeros:0
?
(user_embedding/u_age_emn_matrix/Adam_1:0-user_embedding/u_age_emn_matrix/Adam_1/Assign-user_embedding/u_age_emn_matrix/Adam_1/read:02:user_embedding/u_age_emn_matrix/Adam_1/Initializer/zeros:0
?
&user_embedding/u_sex_emb_matrix/Adam:0+user_embedding/u_sex_emb_matrix/Adam/Assign+user_embedding/u_sex_emb_matrix/Adam/read:028user_embedding/u_sex_emb_matrix/Adam/Initializer/zeros:0
?
(user_embedding/u_sex_emb_matrix/Adam_1:0-user_embedding/u_sex_emb_matrix/Adam_1/Assign-user_embedding/u_sex_emb_matrix/Adam_1/read:02:user_embedding/u_sex_emb_matrix/Adam_1/Initializer/zeros:0
?
&user_embedding/u_org_emb_matrix/Adam:0+user_embedding/u_org_emb_matrix/Adam/Assign+user_embedding/u_org_emb_matrix/Adam/read:028user_embedding/u_org_emb_matrix/Adam/Initializer/zeros:0
?
(user_embedding/u_org_emb_matrix/Adam_1:0-user_embedding/u_org_emb_matrix/Adam_1/Assign-user_embedding/u_org_emb_matrix/Adam_1/read:02:user_embedding/u_org_emb_matrix/Adam_1/Initializer/zeros:0
?
'user_embedding/u_seat_emb_matrix/Adam:0,user_embedding/u_seat_emb_matrix/Adam/Assign,user_embedding/u_seat_emb_matrix/Adam/read:029user_embedding/u_seat_emb_matrix/Adam/Initializer/zeros:0
?
)user_embedding/u_seat_emb_matrix/Adam_1:0.user_embedding/u_seat_emb_matrix/Adam_1/Assign.user_embedding/u_seat_emb_matrix/Adam_1/read:02;user_embedding/u_seat_emb_matrix/Adam_1/Initializer/zeros:0
?
&user_embedding/u_pos_emb_matrix/Adam:0+user_embedding/u_pos_emb_matrix/Adam/Assign+user_embedding/u_pos_emb_matrix/Adam/read:028user_embedding/u_pos_emb_matrix/Adam/Initializer/zeros:0
?
(user_embedding/u_pos_emb_matrix/Adam_1:0-user_embedding/u_pos_emb_matrix/Adam_1/Assign-user_embedding/u_pos_emb_matrix/Adam_1/read:02:user_embedding/u_pos_emb_matrix/Adam_1/Initializer/zeros:0
?
u_type_fc/kernel/Adam:0u_type_fc/kernel/Adam/Assignu_type_fc/kernel/Adam/read:02)u_type_fc/kernel/Adam/Initializer/zeros:0
?
u_type_fc/kernel/Adam_1:0u_type_fc/kernel/Adam_1/Assignu_type_fc/kernel/Adam_1/read:02+u_type_fc/kernel/Adam_1/Initializer/zeros:0
x
u_type_fc/bias/Adam:0u_type_fc/bias/Adam/Assignu_type_fc/bias/Adam/read:02'u_type_fc/bias/Adam/Initializer/zeros:0
?
u_type_fc/bias/Adam_1:0u_type_fc/bias/Adam_1/Assignu_type_fc/bias/Adam_1/read:02)u_type_fc/bias/Adam_1/Initializer/zeros:0
|
u_age_fc/kernel/Adam:0u_age_fc/kernel/Adam/Assignu_age_fc/kernel/Adam/read:02(u_age_fc/kernel/Adam/Initializer/zeros:0
?
u_age_fc/kernel/Adam_1:0u_age_fc/kernel/Adam_1/Assignu_age_fc/kernel/Adam_1/read:02*u_age_fc/kernel/Adam_1/Initializer/zeros:0
t
u_age_fc/bias/Adam:0u_age_fc/bias/Adam/Assignu_age_fc/bias/Adam/read:02&u_age_fc/bias/Adam/Initializer/zeros:0
|
u_age_fc/bias/Adam_1:0u_age_fc/bias/Adam_1/Assignu_age_fc/bias/Adam_1/read:02(u_age_fc/bias/Adam_1/Initializer/zeros:0
|
u_sex_fc/kernel/Adam:0u_sex_fc/kernel/Adam/Assignu_sex_fc/kernel/Adam/read:02(u_sex_fc/kernel/Adam/Initializer/zeros:0
?
u_sex_fc/kernel/Adam_1:0u_sex_fc/kernel/Adam_1/Assignu_sex_fc/kernel/Adam_1/read:02*u_sex_fc/kernel/Adam_1/Initializer/zeros:0
t
u_sex_fc/bias/Adam:0u_sex_fc/bias/Adam/Assignu_sex_fc/bias/Adam/read:02&u_sex_fc/bias/Adam/Initializer/zeros:0
|
u_sex_fc/bias/Adam_1:0u_sex_fc/bias/Adam_1/Assignu_sex_fc/bias/Adam_1/read:02(u_sex_fc/bias/Adam_1/Initializer/zeros:0
|
u_org_fc/kernel/Adam:0u_org_fc/kernel/Adam/Assignu_org_fc/kernel/Adam/read:02(u_org_fc/kernel/Adam/Initializer/zeros:0
?
u_org_fc/kernel/Adam_1:0u_org_fc/kernel/Adam_1/Assignu_org_fc/kernel/Adam_1/read:02*u_org_fc/kernel/Adam_1/Initializer/zeros:0
t
u_org_fc/bias/Adam:0u_org_fc/bias/Adam/Assignu_org_fc/bias/Adam/read:02&u_org_fc/bias/Adam/Initializer/zeros:0
|
u_org_fc/bias/Adam_1:0u_org_fc/bias/Adam_1/Assignu_org_fc/bias/Adam_1/read:02(u_org_fc/bias/Adam_1/Initializer/zeros:0
?
u_seat_fc/kernel/Adam:0u_seat_fc/kernel/Adam/Assignu_seat_fc/kernel/Adam/read:02)u_seat_fc/kernel/Adam/Initializer/zeros:0
?
u_seat_fc/kernel/Adam_1:0u_seat_fc/kernel/Adam_1/Assignu_seat_fc/kernel/Adam_1/read:02+u_seat_fc/kernel/Adam_1/Initializer/zeros:0
x
u_seat_fc/bias/Adam:0u_seat_fc/bias/Adam/Assignu_seat_fc/bias/Adam/read:02'u_seat_fc/bias/Adam/Initializer/zeros:0
?
u_seat_fc/bias/Adam_1:0u_seat_fc/bias/Adam_1/Assignu_seat_fc/bias/Adam_1/read:02)u_seat_fc/bias/Adam_1/Initializer/zeros:0
|
u_pos_id/kernel/Adam:0u_pos_id/kernel/Adam/Assignu_pos_id/kernel/Adam/read:02(u_pos_id/kernel/Adam/Initializer/zeros:0
?
u_pos_id/kernel/Adam_1:0u_pos_id/kernel/Adam_1/Assignu_pos_id/kernel/Adam_1/read:02*u_pos_id/kernel/Adam_1/Initializer/zeros:0
t
u_pos_id/bias/Adam:0u_pos_id/bias/Adam/Assignu_pos_id/bias/Adam/read:02&u_pos_id/bias/Adam/Initializer/zeros:0
|
u_pos_id/bias/Adam_1:0u_pos_id/bias/Adam_1/Assignu_pos_id/bias/Adam_1/read:02(u_pos_id/bias/Adam_1/Initializer/zeros:0
?
u_concated_fc/kernel/Adam:0 u_concated_fc/kernel/Adam/Assign u_concated_fc/kernel/Adam/read:02-u_concated_fc/kernel/Adam/Initializer/zeros:0
?
u_concated_fc/kernel/Adam_1:0"u_concated_fc/kernel/Adam_1/Assign"u_concated_fc/kernel/Adam_1/read:02/u_concated_fc/kernel/Adam_1/Initializer/zeros:0
?
u_concated_fc/bias/Adam:0u_concated_fc/bias/Adam/Assignu_concated_fc/bias/Adam/read:02+u_concated_fc/bias/Adam/Initializer/zeros:0
?
u_concated_fc/bias/Adam_1:0 u_concated_fc/bias/Adam_1/Assign u_concated_fc/bias/Adam_1/read:02-u_concated_fc/bias/Adam_1/Initializer/zeros:0
?
.item_class_embedding/i_class_emb_matrix/Adam:03item_class_embedding/i_class_emb_matrix/Adam/Assign3item_class_embedding/i_class_emb_matrix/Adam/read:02@item_class_embedding/i_class_emb_matrix/Adam/Initializer/zeros:0
?
0item_class_embedding/i_class_emb_matrix/Adam_1:05item_class_embedding/i_class_emb_matrix/Adam_1/Assign5item_class_embedding/i_class_emb_matrix/Adam_1/read:02Bitem_class_embedding/i_class_emb_matrix/Adam_1/Initializer/zeros:0
?
i_class_fc/kernel/Adam:0i_class_fc/kernel/Adam/Assigni_class_fc/kernel/Adam/read:02*i_class_fc/kernel/Adam/Initializer/zeros:0
?
i_class_fc/kernel/Adam_1:0i_class_fc/kernel/Adam_1/Assigni_class_fc/kernel/Adam_1/read:02,i_class_fc/kernel/Adam_1/Initializer/zeros:0
|
i_class_fc/bias/Adam:0i_class_fc/bias/Adam/Assigni_class_fc/bias/Adam/read:02(i_class_fc/bias/Adam/Initializer/zeros:0
?
i_class_fc/bias/Adam_1:0i_class_fc/bias/Adam_1/Assigni_class_fc/bias/Adam_1/read:02*i_class_fc/bias/Adam_1/Initializer/zeros:0
?
i_concated_fc/kernel/Adam:0 i_concated_fc/kernel/Adam/Assign i_concated_fc/kernel/Adam/read:02-i_concated_fc/kernel/Adam/Initializer/zeros:0
?
i_concated_fc/kernel/Adam_1:0"i_concated_fc/kernel/Adam_1/Assign"i_concated_fc/kernel/Adam_1/read:02/i_concated_fc/kernel/Adam_1/Initializer/zeros:0
?
i_concated_fc/bias/Adam:0i_concated_fc/bias/Adam/Assigni_concated_fc/bias/Adam/read:02+i_concated_fc/bias/Adam/Initializer/zeros:0
?
i_concated_fc/bias/Adam_1:0 i_concated_fc/bias/Adam_1/Assign i_concated_fc/bias/Adam_1/read:02-i_concated_fc/bias/Adam_1/Initializer/zeros:0
t
expert_weight/Adam:0expert_weight/Adam/Assignexpert_weight/Adam/read:02&expert_weight/Adam/Initializer/zeros:0
|
expert_weight/Adam_1:0expert_weight/Adam_1/Assignexpert_weight/Adam_1/read:02(expert_weight/Adam_1/Initializer/zeros:0
l
expert_bias/Adam:0expert_bias/Adam/Assignexpert_bias/Adam/read:02$expert_bias/Adam/Initializer/zeros:0
t
expert_bias/Adam_1:0expert_bias/Adam_1/Assignexpert_bias/Adam_1/read:02&expert_bias/Adam_1/Initializer/zeros:0
p
gate1_weight/Adam:0gate1_weight/Adam/Assigngate1_weight/Adam/read:02%gate1_weight/Adam/Initializer/zeros:0
x
gate1_weight/Adam_1:0gate1_weight/Adam_1/Assigngate1_weight/Adam_1/read:02'gate1_weight/Adam_1/Initializer/zeros:0
h
gate1_bias/Adam:0gate1_bias/Adam/Assigngate1_bias/Adam/read:02#gate1_bias/Adam/Initializer/zeros:0
p
gate1_bias/Adam_1:0gate1_bias/Adam_1/Assigngate1_bias/Adam_1/read:02%gate1_bias/Adam_1/Initializer/zeros:0
p
gate2_weight/Adam:0gate2_weight/Adam/Assigngate2_weight/Adam/read:02%gate2_weight/Adam/Initializer/zeros:0
x
gate2_weight/Adam_1:0gate2_weight/Adam_1/Assigngate2_weight/Adam_1/read:02'gate2_weight/Adam_1/Initializer/zeros:0
h
gate2_bias/Adam:0gate2_bias/Adam/Assigngate2_bias/Adam/read:02#gate2_bias/Adam/Initializer/zeros:0
p
gate2_bias/Adam_1:0gate2_bias/Adam_1/Assigngate2_bias/Adam_1/read:02%gate2_bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
x
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:02'dense_1/kernel/Adam/Initializer/zeros:0
?
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:02)dense_1/kernel/Adam_1/Initializer/zeros:0
p
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:02%dense_1/bias/Adam/Initializer/zeros:0
x
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:02'dense_1/bias/Adam_1/Initializer/zeros:0
x
dense_2/kernel/Adam:0dense_2/kernel/Adam/Assigndense_2/kernel/Adam/read:02'dense_2/kernel/Adam/Initializer/zeros:0
?
dense_2/kernel/Adam_1:0dense_2/kernel/Adam_1/Assigndense_2/kernel/Adam_1/read:02)dense_2/kernel/Adam_1/Initializer/zeros:0
p
dense_2/bias/Adam:0dense_2/bias/Adam/Assigndense_2/bias/Adam/read:02%dense_2/bias/Adam/Initializer/zeros:0
x
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:02'dense_2/bias/Adam_1/Initializer/zeros:0
x
dense_3/kernel/Adam:0dense_3/kernel/Adam/Assigndense_3/kernel/Adam/read:02'dense_3/kernel/Adam/Initializer/zeros:0
?
dense_3/kernel/Adam_1:0dense_3/kernel/Adam_1/Assigndense_3/kernel/Adam_1/read:02)dense_3/kernel/Adam_1/Initializer/zeros:0
p
dense_3/bias/Adam:0dense_3/bias/Adam/Assigndense_3/bias/Adam/read:02%dense_3/bias/Adam/Initializer/zeros:0
x
dense_3/bias/Adam_1:0dense_3/bias/Adam_1/Assigndense_3/bias/Adam_1/read:02'dense_3/bias/Adam_1/Initializer/zeros:0
x
dense_5/kernel/Adam:0dense_5/kernel/Adam/Assigndense_5/kernel/Adam/read:02'dense_5/kernel/Adam/Initializer/zeros:0
?
dense_5/kernel/Adam_1:0dense_5/kernel/Adam_1/Assigndense_5/kernel/Adam_1/read:02)dense_5/kernel/Adam_1/Initializer/zeros:0
p
dense_5/bias/Adam:0dense_5/bias/Adam/Assigndense_5/bias/Adam/read:02%dense_5/bias/Adam/Initializer/zeros:0
x
dense_5/bias/Adam_1:0dense_5/bias/Adam_1/Assigndense_5/bias/Adam_1/read:02'dense_5/bias/Adam_1/Initializer/zeros:0"?)
trainable_variables?)?)
?
"user_embedding/u_type_emb_matrix:0'user_embedding/u_type_emb_matrix/Assign'user_embedding/u_type_emb_matrix/read:02user_embedding/random_uniform:08
?
!user_embedding/u_age_emn_matrix:0&user_embedding/u_age_emn_matrix/Assign&user_embedding/u_age_emn_matrix/read:02!user_embedding/random_uniform_1:08
?
!user_embedding/u_sex_emb_matrix:0&user_embedding/u_sex_emb_matrix/Assign&user_embedding/u_sex_emb_matrix/read:02!user_embedding/random_uniform_2:08
?
!user_embedding/u_org_emb_matrix:0&user_embedding/u_org_emb_matrix/Assign&user_embedding/u_org_emb_matrix/read:02!user_embedding/random_uniform_3:08
?
"user_embedding/u_seat_emb_matrix:0'user_embedding/u_seat_emb_matrix/Assign'user_embedding/u_seat_emb_matrix/read:02!user_embedding/random_uniform_4:08
?
!user_embedding/u_pos_emb_matrix:0&user_embedding/u_pos_emb_matrix/Assign&user_embedding/u_pos_emb_matrix/read:02!user_embedding/random_uniform_5:08
w
u_type_fc/kernel:0u_type_fc/kernel/Assignu_type_fc/kernel/read:02-u_type_fc/kernel/Initializer/random_uniform:08
f
u_type_fc/bias:0u_type_fc/bias/Assignu_type_fc/bias/read:02"u_type_fc/bias/Initializer/zeros:08
s
u_age_fc/kernel:0u_age_fc/kernel/Assignu_age_fc/kernel/read:02,u_age_fc/kernel/Initializer/random_uniform:08
b
u_age_fc/bias:0u_age_fc/bias/Assignu_age_fc/bias/read:02!u_age_fc/bias/Initializer/zeros:08
s
u_sex_fc/kernel:0u_sex_fc/kernel/Assignu_sex_fc/kernel/read:02,u_sex_fc/kernel/Initializer/random_uniform:08
b
u_sex_fc/bias:0u_sex_fc/bias/Assignu_sex_fc/bias/read:02!u_sex_fc/bias/Initializer/zeros:08
s
u_org_fc/kernel:0u_org_fc/kernel/Assignu_org_fc/kernel/read:02,u_org_fc/kernel/Initializer/random_uniform:08
b
u_org_fc/bias:0u_org_fc/bias/Assignu_org_fc/bias/read:02!u_org_fc/bias/Initializer/zeros:08
w
u_seat_fc/kernel:0u_seat_fc/kernel/Assignu_seat_fc/kernel/read:02-u_seat_fc/kernel/Initializer/random_uniform:08
f
u_seat_fc/bias:0u_seat_fc/bias/Assignu_seat_fc/bias/read:02"u_seat_fc/bias/Initializer/zeros:08
s
u_pos_id/kernel:0u_pos_id/kernel/Assignu_pos_id/kernel/read:02,u_pos_id/kernel/Initializer/random_uniform:08
b
u_pos_id/bias:0u_pos_id/bias/Assignu_pos_id/bias/read:02!u_pos_id/bias/Initializer/zeros:08
?
u_concated_fc/kernel:0u_concated_fc/kernel/Assignu_concated_fc/kernel/read:021u_concated_fc/kernel/Initializer/random_uniform:08
v
u_concated_fc/bias:0u_concated_fc/bias/Assignu_concated_fc/bias/read:02&u_concated_fc/bias/Initializer/zeros:08
?
)item_class_embedding/i_class_emb_matrix:0.item_class_embedding/i_class_emb_matrix/Assign.item_class_embedding/i_class_emb_matrix/read:02%item_class_embedding/random_uniform:08
{
i_class_fc/kernel:0i_class_fc/kernel/Assigni_class_fc/kernel/read:02.i_class_fc/kernel/Initializer/random_uniform:08
j
i_class_fc/bias:0i_class_fc/bias/Assigni_class_fc/bias/read:02#i_class_fc/bias/Initializer/zeros:08
?
i_entities_emb_fc/kernel:0i_entities_emb_fc/kernel/Assigni_entities_emb_fc/kernel/read:025i_entities_emb_fc/kernel/Initializer/random_uniform:08
?
i_entities_emb_fc/bias:0i_entities_emb_fc/bias/Assigni_entities_emb_fc/bias/read:02*i_entities_emb_fc/bias/Initializer/zeros:08
?
i_concated_fc/kernel:0i_concated_fc/kernel/Assigni_concated_fc/kernel/read:021i_concated_fc/kernel/Initializer/random_uniform:08
v
i_concated_fc/bias:0i_concated_fc/bias/Assigni_concated_fc/bias/read:02&i_concated_fc/bias/Initializer/zeros:08
m
expert_weight:0expert_weight/Assignexpert_weight/read:02,expert_weight/Initializer/truncated_normal:08
Z
expert_bias:0expert_bias/Assignexpert_bias/read:02expert_bias/Initializer/zeros:08
i
gate1_weight:0gate1_weight/Assigngate1_weight/read:02+gate1_weight/Initializer/truncated_normal:08
V
gate1_bias:0gate1_bias/Assigngate1_bias/read:02gate1_bias/Initializer/zeros:08
i
gate2_weight:0gate2_weight/Assigngate2_weight/read:02+gate2_weight/Initializer/truncated_normal:08
V
gate2_bias:0gate2_bias/Assigngate2_bias/read:02gate2_bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08
o
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:08
^
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:08
o
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:08
^
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:08"<
losses2
0
loss/log_loss/value:0
loss/log_loss_1/value:0"
train_op

Adam*?
serving_default?
)
u_type
u_type:0?????????
'
u_sex
u_sex:0?????????
*
u_org!

u_org_id:0?????????
'
u_age
u_age:0?????????
*
u_pos!

u_pos_id:0?????????
,
u_seat"
u_seat_id:0?????????
7
i_class_label&
i_class_label:0?????????
7
i_entities_label#
i_entities:0?????????@,
label1_score
ctr:0?????????0
label2_score 
	cvr_ctr:0?????????tensorflow/serving/predict