??
?+?+
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
	AssignAdd
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
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
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
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
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
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
5

Reciprocal
x"T
y"T"
Ttype:

2	
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
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
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
shared_namestring ?"serve*1.14.02unknown??


feat_indexPlaceholder*%
shape:??????????????????*
dtype0*0
_output_shapes
:??????????????????


feat_valuePlaceholder*%
shape:??????????????????*
dtype0*0
_output_shapes
:??????????????????
h
labelPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
j
dropout_keep_fmPlaceholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
l
dropout_keep_deepPlaceholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
P
train_phasePlaceholder*
shape:*
dtype0
*
_output_shapes
:
d
random_normal/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: 
?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed?*
T0*
dtype0*
seed2	*
_output_shapes
:	?
|
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:	?
e
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:	?
?
feature_embeddings
VariableV2*
shape:	?*
shared_name *
dtype0*
	container *
_output_shapes
:	?
?
feature_embeddings/AssignAssignfeature_embeddingsrandom_normal*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
feature_embeddings/readIdentityfeature_embeddings*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
e
random_uniform/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
W
random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
W
random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
seed?*
T0*
dtype0*
seed2*
_output_shapes
:	?
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
u
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*
_output_shapes
:	?
g
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*
_output_shapes
:	?
?
feature_bias
VariableV2*
shape:	?*
shared_name *
dtype0*
	container *
_output_shapes
:	?
?
feature_bias/AssignAssignfeature_biasrandom_uniform*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
v
feature_bias/readIdentityfeature_bias*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
??
Variable/initial_valueConst*??
value??B??	? "???1X=:kx??'޼%??A???>:y?=??7?s\?=ʈ?蟼??M??^]?b????DQ??)???1?T?X?Q??ϝ:??I?<}??=?H??U??=/R7??!ུ??;??>?>HU?=
>l^?<?`?S	??'?q>??=?^<<&??7?=?w<~o?N?<Or???U;?w]??NS?*0=o?ۻ!b?<їc????==??=?8?=?C=???'?h??? ???-??<?? J=?3???[????=?mf>???pNU???|???? ?6??<$|;?k=??Ea?=eΏ???`??????an?գ?<?y?? ?ӿ?=????8 ??!?%=Q?>)?ν@?"???<?}?*?	??|?:׋?=??<?MH=R?c<???<?~ռ????q??+p>??ͼ1????????e??L)??:??+	?0A>?>?5??\??<K??<?,輾p??!?=?6=?V?????μ???<G????n????t?-???>?=ķ=2?
? ?ý?K?????=\?5=̫=??<=|?~=??K????=?????P???N?=Zb=???=Ŭ???m??*?=G;z=?I?:=???>??qe???|$=?rs=JV?=y????*????W?l?s??>(?=bdt?d7?:]=?lk??r?!?zĂ=???=???=5??=T?=Π/=.??=y??????M
?Z2n??mH; ?2>C6?<0z????ν?ת=??R?L??<M??*>xFT=??w???=7??=ʷM?>??:?+?tk?<G?????R>Ȱ>?X?qa??ּ=;]?(>>?=?]=?[????>:l۽"߽??s>5g?#?=???=z??=??v=C?c<`락5??<?H???ä<??T=x8??[?<?'???s????=?5?=???<?ߌ={d=????EqM>??=K??q?N>??F?՛&??8?=??:?,;??E=??=?ԩ<k%?=?!>??n???Ьf???<W$?=?[?;;??=?A?????T
۽???=?ƃ<g?=?C,;?#??qͼ#??=?Dp??????=8?=K??=\M?=:?9=n???B????*@???????)=A3=l?>V??< ?V??4?????ci????)=?s????^??l?=a$???8<H??<_47?^v???m=?\?=?Խ?4???b?=u] =c????`0?l??::???l??#3+??C)>???<r箼(Gh?$Fļ?=	Z??L?=????h?=B?Ƚ????N???????ݽ?^??ŷ/>-??=Fa=???I>u???i?=???v妽mI?=???<?~?о?<?Ȝ=M?????<?`rM>оI???F?֔?=b}?E??{?L??9?<??? <?h???[?鵹=?n???ͼʍ ?h?@>??,>Ï?ȳ?O??=c1?i????p??????E???u???3????<?	?<'4?=?<<???U??<??????P????=???<?d??c???=Љ??r?????=Gx??????~f?=??4="??<d̽?
J?3o?
??Ę5=????M?<SJ?d??=]??=H??=?>??=h???W1?Bi?=L64>?`?????????<?M<;??<?F??-?<?B?<???T?=iʛ=???8??#??=ݜ7>?U???u=)-^??K??=ݽ??=?
ѻX????t
?V2?=?퇽?Sv???o;??=?.=>????1??<ο*>??+??j?=?#???쇾m阽g?ּȩ>?U?????=???=?>?I?=??=	u=???=???K>g<SK?_b?=	????:=??5=?>?Nd??.????8=??ػ???8R????<Y[<=ad?<a_??$=i??: >R????~?G+?<???<?a=?м???:??=?W?>?Di=?$ܽ??9.??9 ?D??<?h???t=?????齚'???b<?ν_?]??j?;?,Q?P??=?uü'^=+??<??<@??)6??Z????=?
?=?j>y?ٽ?
????VL?=*:?<d_?=C?*??P8>?!????<$??o
>9j?<?#>?????役?<?`??M0>I??=?>F???S	?=~??=????U?;??ػR????;?t=DU?<???~X>1?M??8`>???=n?K=n??=
)y=<Ӑ?A?u=?J?Ĕ>@?????;Ɲͼ>???_????;???V=?;>??+?1G>??2<??????0>ٙ?=t؈??ځ="7??????◪??i%<?D?<!?>?>>/?3=?d?Q.????`????6????=|??8?ֽf
??wd???J&??[?=??=Ρe>?3???ߌ=??>?J>?ݽp9?h*?
?=X???;>??ý??N=]?=H_4=?A=???T I=	#??ۮ????=??=?n?<3???ȽMr?\?????<^???Z???=7?=3Wڽ!ָ=W?3>M?>?4?<7&+:?+?9???yy???4=?-ǽ??e>?[??????`r?=p?>W?N?9?_???Q	?!ú?й<?{?<??)>????,<М??bR??k?^=9??=?ȶ??U?=Ǽ?= ψ;˶d<?~?=d	????k?<`??????4g#=n὞6???6'??4a????<?*2??ʭ?????>?^??̂?5/?=ƻ???/????h?I>!f??U=?.B=8B?ø???q=?a?9k??_? ?:??<?d??Q5P>F?7<g?/=hP??4?=???=?^?̍?;??<1ξ=Wň?u??=`??????<"\[?Tᑽ???=???=/?a?h?>?y=A*?:?,4??T<̕=?X=?9?<???jjT?a^?=?????}2=???<???;??>??P=??=F??<????z?f>?q?<?.??? >??>?Y??L<G?&?a=?*5?JMS=|n=??=?q?<(VB>?b%>7)<?M?????\1????>?`???$V?fɈ=??>?ԋ????<Ә???+??mݻ??ڻw4??5?Ͻ?J??a?-????=??????B??1????=?h??\?=T?=?|=%?=N:?=??%=h$`?ս?Z½?ޜ;b????Ͻ?2r=Y?o???B????=??I???g=į}=[??r?(=?Vs>?Yj>ADw=?I??{??=g?E<U?=?#>v">d?j??G??qC???Լ?_?=.???2?=Q?=?qսy4?=賈??????c????>?ɽ6涽??A????=??M=??+?S??m??^5?XTs?? _=a?;?i?a??=??>˪??d??<??=?8ѻ?C>@?I=L=?[?1?=?W??[<??????=t????K=?d?=]O8?6?D???=?a?=>??<??)???[? Ȏ??w?<`?u?????;?	?<ݒ?=`j????L>q{?=쿣?6?????;??E=??]?????_ٻ8L=???w??7?_??=qL?<??"?Y9?=K??=???)N>?&??i?սG.ļ
?<w?Ͻt???ɼ?Ž?$?<?6>???!,>?Y%??H?=??=ص[=???=;????K?,???Z????7?????M????X?@=??ֽl?=?a???Խ폨??*p=߭???>I??a/>?ڮ=$<>?=;?ȭ.?4W?] ?<O 5???>??*?|?= '@?uN??ny>?x_= ??????=?|>? ??j䰽%{?=f??-?>>?H<?i=??<s?>?/<<m)=y/??}b?????<????}?<hN>?,>?<???X?0??=D?=?ڹ=9#?=L|v?mb[=,D?>??潁\???O?vN?<g?=6	K?????{=?,??B??
???>????Q{ >)?>q >????!??=|Z/???^=ϡ?=?$?=9pZ????=?E>.?/> p=|\<ɽ۰?<񚅽e@?=??}=?wܽ??M=?@?=
???\??:S??<)(@=rc>???^n<#=>?$?=*?ݽ+?@<?^=?U0??[?=??J=D?????	=?<?*??n?i??2"?E˽Bs???;??<>?=B???dt?=I?;???=??E??b=w?6>??$?}?<5?U?v????AT=5?????<?(?y?>۷?uY>????~ ????z\??|NB<??/????l??x>a?:TM?=J??ǪC???n?A,U?p?սK?b<zl輬?	?jT)??e????<??>?i~=)?'?sk????=|????@=.i?=^??=???<Nk??bc???>??x??dI???޽k?	??R?n?Ѽ??R???;\?>y֛<???押??>?^>]2?L??=պ???=??<e<?=H?=??=`;vǽ??<%? ??(=?p????:??8?h??=?q8=??=4????׽??j??F>?GD<?H?=Ť??@.`?)??=??=(ᔻ???<??)=1?<>U?ߕ?=PQȼ?7=???yt?$?Y<????Zδ=+i???X=C?h=?+Ž?>`p?<?>?A?<?#0>?@???,?=??M=?5m??ڿ=?6U?:)?=t?i>
???????u>??ʽ??6???a??ܥ=J??=????v?=??k>/??-<?żv2U????=%??????X??T??=u?>z;??c?]=???<?"0?A@˽H5??vc=?=??*?><<-?>T??IK>T???w??G?7>?Ǒ??JA=?
?='??<???<?J?<?? ?`ޅ?˺ɽ??????=e׼o&?8j;5X>?=1??xP2=??
??l>?H????v;??۽;?)??H??8<?ʽ??
>P?ܽ?<=m8????d=????2???a[=J!??b???[;+??5>?:??Cr????=??9>??g????<??۽?в????=??g??s=????n?*??:?ӎ??1=?Ž?C???:V?????	??Bͽs? >??????<??=?C>??/=?콿,7?{D??ޟ>??k=????#?޽ (?RqE>c%}?Io?=R>AP?<???<?^¼C=??k=????Q?<?PJ??;<>(??=L?>G3۽yD??ҽ???<h?ܽ??9?N??<???=?3???s?=?ג?
?g????h8??O&<?Ӊ<W?=E?(>HF??@??,??-??<?v>????麽f??g ??>?׽ `=&????;??ph???<v????~=N?彧??EIսI?н?WU>??%??????=?k̽Ɓ@?#n????ĽL?H?s?ڽ????<??=?3?5??>?Y?=^=!=1b??=%??=K7?=a?=?ʳ?????H??=?c?=D???N?=cʚ???5=D?}=	?<~?=T-??5HY????<.?׽?=??<??%>DVt>???=??5????<"-z;?V
??3.??0????=?;???+?=??o>?c ???=.>???e??'?@??ڼ?̿=????׳?<P?<w?=G?Y?,??>?(H??4Z???<???=??9?O?t=?.<????e4?????޽??F.b??*?:???????i)?2??lǹ=PA?>?}{>8z>?04>.u????=?3>F?=?l??g???j?=@?һ???=.ԉ??P =??T?3??=&?<Ue???h0???Z=??I<i???gy??k⽔+ͽ8???%>e>2????k?=Ȱ?? ?	???'>]?=?4?;?8\=\?d>n???py<D^Y=M???q??p9<?:?s??? ??<?}???|<????M??<L%A=1?=????^??=[6????=?̓=?L?????=^)ӽ?? ?յ??B?.?=?мX ?;??&?1??=??}=?D?<8q?<?? >????E?=??????5?}?9p$?b????<???=nO	=ꉤ?@??? ?=ƛ=??=|? =?ý?y<??k.9>?>??[?J =??R??'=?'?^????????Z?????'1??????9??bq???d????">??:=
R??G??K???????mIN?^?<<?(*?Ƣ<?I<=?g???zݽS???	=?4??;$?<??G=?0Q=UΓ<?X,=:?<('?>?*=R??=?x+>?Lp<?=2!?b??=|7<e?
>?????Z7?=%{=???=$???????Ξ=)T>?h@?e?)<U????z=??d?A?i=????1ka<????u̞?`??ʠ???~? T??????Еs?+?<,s?<?I??>u?<?_??+=?5?"ɽ,L?R+???jI???=6>?8?P?}??PI???$={F+?TA"?????SF=???:??H?`?<??>??=*??<?-?????<Nn=w????=?6??l2???n?$E<g?x>a?=??;???=L+?<?7n=???=/?>(w?<?%\=?G??ӳ?06d9??<K(G?3?;???<?Z?ܑ????=??<??E=5l?ۥ<16?<8??;,X???=??<??G?j??=?N? <F5??I6?~?=(??;Y(??)? ?=?s@?E? ??& =_j???T\>8j#=?6?=??ý???????=c???;ۅ???????ع?ٸ<-??=b<>??????;&?=?A??/0?<tqf?DȽ6A??yM׻?K?=???Mo/=?"z>wK?<E0I??ݸ??p?=?;?2J>??= ?%=+??W=c=?2???ar<(_?<=?<???=??$?A4=_Z3?eh???LJ???'???/??+7>?j:>???????k>Ğs?Ϭ?=??)????:@<=?Ѝ???!>[J???]ν?Eu=n??=?K?=???J,?2r?=
?????yK>m>T,??XP?@??????D?߼????D1ν7[?=?U?G?e?^oj=??=J???Y?y??.>Kn?=NCN>?ع=?Ka>¦>]@K???ý???X?½G???!?
>p?>=x^	?WE?7B???ý_?<@F>?Y5?1??=?????ӽ&?ݼkH??0د????o2=??=ۡC>??<??O??6??zv??VP>7??:??<????ZB??*'>{-?????=??X?A}??\=??>F??T ????	83>=X??wo?ܒ?????<1?(???ʽRݽ?6?w"????<??=A`<????觽~?߼%7??P?Ӊ	?^ꬼ??=?[h?W?,?3?<?d???Q??g??c??????@?W? =??ܽ'?=B?	=?????B?<?S?=>?ƽ#????a=??????J5??ǌ?=??=wKM?'C>?2??-?<?Qǽ??=?&>?Ɓ???=?zV??3?B%?=\ O??X>|w;=\?"???;F5???%>X?<?c$>??-<???=?+V=5?ѽP?=;?<?s??a??Ց?????	2	>?[?<??????=?Q>`?x??=?????~	?>????l2<q?#?lҥ?A??=??
????<n{?
?=??:.3?=)9k<4"???????_=t?2?]ƫ=?B!?q????????=?????????R?<t??????=?m??;?t????
=ҽU???^:?Dc? ?9?od>??K??ռ?:@???;8???/???X??u? =0?=?w??=??0?{? >]??=<??=n?h??yὒ????'?͢E?????????????!:?9?=????Y?????=?????L????=?=??=F3??f??o???=Yf???S=??5?R%>?sM?~
>?Xj=??k?e?1=??F?ā?????:R$????=???a<L?#?<???=?=9??:??=??e=ˁ???	?<??5=Z?Ƚ?U)=??f>&=???=??=?n?;6?#R???3?<Qw?8??=??#???=ƗE??"4?;{?=?.(??????X<=8\>?=?2>`>?~?=-L?=??x=????;(??< ?5???p½ީ?Y8F=t?f=?H?=?????=\???? 鱻???=?TR?????'>)?a???'=2]?=
?C?R>???=q?????=?Ŕ<	??=	q?>Υu??????;?=?=buE=??=5?<~???'>ѽ?->??=???=_??<????X9????0=?`????;+?f=:??=??=???=U?X??=c??<\???c?=? {??;X??@<?P>=???;_N???O-=p?޽?+?<Y???Y??=????]?L??Yl=Fҽ???7?#?W;=?䭽^ ý???????;??=?]u>??1|=???p?|????؊B?/?<E?޽#?^>r?Z=f=??h?!??<[?># ٽ4#?????`?_?<>C-???R=?e"=ٿ?;???=&,?=?????L>?@;???<??6??=???<4?켤C7?? ???&?<??z??bz<???<?om?o鼏??=?s??];a?=?~?????=H???M=k??=B:r?L$>??=??=????ǀ=\bm??f???>J<?=*?????J??,#>1k׽?<????r?=?????B?=^=2????xQ=+Ǳ=?:5>Ԉ????='Z????<<???<??W??zy??g;?h>?M????????=7?j????@?g?l.S>g??=?????=ԭW=?	<?>iP???=qȲ?l??=Q??????=????j]?=??????Խ???<"Ƽ???/????????0<X??=???=c<SΡ?=?/>?? >jhn=?}&????%;???$?=?>ÊƽO[R>L???z???An=??>?? ???i??o??l????խ=??5?Գ??ͽ?h?<?&>Ճٽ??&?8y/? G?=D(?<?A??Ϊ=???=)?7??]???z???=W??	Je?????b\??p???h׽?1?=Z>dP??'=mq=?8ýU?C<n??=!>h?9???=?$U???=?w?P??<?=?<???<??`>??`=??7????=????T?D????????????<?u?=????X<>?5??f??ӫ<???=???<????=]E?<??m>?-=]?????	>???=??_>?>??N?6?<?E?N=????f?q?[<??=%t??,?=sw?=??"=?彴?=?+?=?~??t??E?a??)Q?p???Z??<??ǽ?$>?>??۴=\5??|?>? ??Cۣ=N?N>?y=0?\?????[??h??>??>?D???w??A???#mܼ7?<??$?N?=??c?<=|=? ?ɔW??J=??/??????V?O??W<?0???ޕ>f҅???@=.p"=?x>.????"?=*?????輡?/=jYd=???;:?=K?=?#9???*=)ƽ???<?A1??D?=𪽸-?\?'=ҵ???qt?????D?;wD???????>????N?K?l?<1?;eS?X?>?_ ??	>?HS????录H~=????f ?7T?=?V?=?d???O?'3??&???gZ*?J?y?????އ???15>??#=?[???T??<3?=?(?<LI<??C:??????????d=?=?=n??;?tp?;=??߽t-?}0??G??<bk =}??=??=??X=kuz??@J??@g??7?=??;OA?????=???>v??????=&???X?=?8|=?????mz?i=?˼Y?Q?|?<???<}e??X?????=r??: k?>!M[=?? ????=L??F%??????	=???=???|H???={???hٻ?D?|????$?<??<?6??a`k=4?<r?????y=?\?=I{>j?{=??=3 ?C???q?߼???@n?=?;??=?U<????7z=?S?=?񤻾??^ؽ4(??	A=??<m??=*{??VgŽ?n?9?z?????<?u??A,<??w?<???|2??O¾=?U???O>???<??=???=6??<?B??	?ќ=?N??켦?7:ѐ>?????Qֽ|]?U??i??=?l?=???<?pl??4]????d???n?3???F?Ż
圽??N=??;??e= ????>??ؼo??=#:@>؋???????{=??=??????;?=B? <Cw>"ѳ??$??%?=??u=?q8=,??=?:???U=?ì?X?|=?"??m\??.>??->??=ר?f?t?3G"?&b?<?R???U???$=?'?;(?H??l?=Z??=?N]?&??TI=>??<??a?8db??%?=v⊻??=?߲???y>Rd?=&??=?>???9?=_?9>?"	?/ň<x???!>L,??1????0=׌???????>?Z??\?&????\?<???F?n???#??Q???F=??6????;?~?
????;??^=?e?Z?=???????Q>'C?????&?=tL/???>)f?ú?;~??L???ޮ???`??!???3>??E???D??l~<??B=^??=:oz?ƒ=????ë?"??(ǽ??c??d?<?z?? ??????%?2> ??>?Q`=01?=??޽??=m>{I?<??%?SV?=N?b??Z=???r???S??<!?л????(?1?-???8??6???6+?<q??=e?-??
==?e??=~+???B?=q =FMO?=??<]??=?w?= ??;̮???=L%?=???
>۽???=??ѽ???=H???8?{=v??=?%j?>C=?>py????T??????J??6??<??< ?E>}?X??w?=cJ?;\??8?ͽ?L?=?=d??όf=?׽???_?y= I??Q?¼?>????;#??2???D??nD?=?O4>???=?=(???Լ???=ciҽ]ۜ;? ?<<????¼??>>߉ҽ???=:??<Sp>? n>b)E???P=< >?????`<?+?= B%?n??9I?=?[?=
?ý0E?=???p?????=????6??????=t?j?,??=??=Y?<"}?9S`???&?\	?={?Z=H?>ߖ???N??o?????>?RT=p?R???=?b?=????-U?<	?Z?{?;2?=`??=??=V??=@?<?#?=V?B>???<???P?h?????????>T>?>???ޡ???a<8?>?>?׆?y'˺??[<P?(>V?????????=??=T?<M╽#>X? <?hv????ݽ???=?O????&>?u>=???<??@?w?)?ۇT??'????'>v?<?O^?bN????I???/?ڊ?????=???=a??;??6<?}??ub??<?=??X?s=Ѣ??*?=??;??P?0">OG?=?Q=?/F>?C?=-b?<D[i=?&???ϩ?ķJ<Um?=????a?#??&?aY??????;?=?4?=???=l???T?=?ii?Aʽ)d	?$#=`Z.??%E>R?F=<??a?/>?ȏ=??<HB???? ?B?????<?ѳ=??Ƚ???< ?q?=???`?=?k???)-=?\|=鸌?N?ռ??????<m?:=?)h:j?=????ߗ?SF????;SMZ<<?<
 &??<???w?|Q???d>?{=?H????ô?\.u<:+?<? ???<ꭌ???d=?mػ??>?<?=a??=IB{=e=?=?`??2`??p??˂=???9?<sx????)??G?<?.??Y????[???4=  >=?K?>???<?нkد=k?="z<?{Ӓ??
o??????s)?F????ܽ??o=??u=?s>??<H??C??<?>f?M(
=9W???&??׮=)???è?%??=O\1>ʍ????[<??R<?v?=N?K=A3	??t?<w|???G?=M?d>9^?=??>?Bн??-=!Q ????CΊ<\?ӽKB?<v?ϻA㓽HyL?Z0?<rY???^=??=?=C*=#X?=?v??ڛ=??{?	?ݽ?6>?[X>݁R>]??=?????/>AB??m??<?ޝ??"?i??<??5?????>???N?<?<?=??>}?н?#???w????%<??'=$/??p<+?>ON!=??<= W??
???>??? H>a?>???TW??"?>???=??7????=?,?>??̽{?<???G<S?????<.?ܼ1??O?=d_??1??O>??X?j~=? ?;4l?=?'=?????_?v\>r?p???<0>?0=??<X0?=(y<>?<x5????;?H
=??r=???<7D=??j=I?=Q??=??Z??;=>ֺ?6?es?=?U>?-???=?۠;????_?m<Km?<+???	?=z???<?>?%???Z?P?=Ϗ	>????????^????{??=ep
=)֮=?=>??<?%ټQUP?}D?>^V?'????B>???=?o5=??=??>m?$?'?V??fL?7>?^??;Z=J?<?%=??=?wM?ό=`\>?;??%kC??????;m?;?*>!0???B?@???f??=M?e?p>[??=?/=?h?<?m)>??I=?}g>=?μ????=??>? =?>;\?q???=?=L?Y?=???=??>*J=?<??}?=??޽?v??T$>)??=?ӵ=;??u;(?1y6>?????XU??C??W??<???=?Cx=P$e?8*???P????Y>?|>??*??t?<?ߪ=??????<???=9?ܼ߾?<??B=h?弿q??????P=??սp???x:<??{>?g ???;???<??=~??????.s??qTG?Խu?=??*??ޚ?????0w¼@?2>??????>???ұ??i?=??=ȁ=Y???zI????<?f=i??sCl>?e?<d ս*;????W??<?Bɼ???=??<2??=/?F=3?=????????=??=?R<*??=?#P>C???/>m??=j???_?Ž][h=?:?=(X??ԁ?]$????=w?;2>rs?????)cǽ?R?Z??:ΙP=-?6???*=Y?????<\k9??}?=Ҁ???A??=I[8??;>??d=Y??<]=>ў??^n/<s?3>O|$=?q?<?<=>????
j>f_?<???=?t'?1?@>?????^??(?=҈??a?R??S???.?%???M?=X??=k???=F?P?[?K??=@m??"YY???:=?ɲ=??|?T/>=?y??h??HUŽ?~?v?=,˟=="?<N???????}????????RK?yv?<:F?=U?|=gd?<??5???l?<␸=????tA?=_Xa?գ=a?f'?;1D?S̻a??R?ƽ-s?_S@?l??_J軒d?қ????þB^??K?m= ?0??? ?&A?=???=????=]?S?=dO?>s??;?W?=U?8f???iJ>G=????=?
_???t??ǒ<??S};?ʼ=6<G=???=S?0>?r=]5????<?E???F=0{?=??(>???`5?=Ӏμ>??+Ƚ??=?[?.?y8G???<??ɽ? 	>??˻ ?=,S?=C??@? >s??=?c??&ɑ<?a9?jk?<ےW=?ļ?8?=2?=E??V9?=????& ?"?>uo???m?=o?<???ᥘ<7P?????,<
?=?z????Nn?=	˸<G?{=? ?=/b?<?D???m=?????7??u?;??t&?=?4|??`ػ?U?OǽF??jFE???=??J=?҉<0驽???=?䞽??=?,??n'Q?목<??L=a_,>r?
??@g???wn? k?????>??;??<??;??S??58h=Iq?=?Ky;t˺;?E?1???^??=??<?`=??D>m??=??!=??>?|???(??0?n??+?5?=?
???т=?%=?{G=Q??=?&?=>??=Bu ?&m/=`?q??[2>^F?~Oʺ?8?<,?,ڼ?&\?W??}??=?򉽪L??*?2=/"?={??_?F?7?;2_˽!?G<~?6>??U=??%????=j??=ۣ??ñ?SM
????Esn<???=F>???s????=?r??k?V??p????=????k????>?ۇ=?ʽ?E=Q?2=???:??4=	Sȼ??0g?=???=o??x?~? '?=??>n{?<q?5??z?=??Z?E	?K??
???\?=]??<uuнI????=G???3?<?$??Â<v?$?W?;S?ӽ?g??C??<҄9?6V>xoH??i?=
r??D+J?<p?>pj輚???}???g?/=?~>?A?=&???/?"??=[j????=?e??ay<`???>.=?Y2?p?r?7???޳?f??I??ED?=H?=ƇI???=??<>?? =:??>/?`?DK???^?r<????C?:???<?0z=?-?<?=???=???=??^?%̇<???=?9?=UX????7?f?=t$=??}=mҹ=?]!?a災~p?ܛȼ??(?=?޼??=??J?!B?;Zs"=?_>?>|L?=?~ｻ?Z?M?½ӠP=S?7?S??<?㘽t??S½$tT=?!̻e??>Z?E>x????z????1=?$? 3?<N??=?0?=<%?=???<?j??7??=-?ڽ?b?V???+?L?<9???;Z?	н%?˽??=Lb??&>?p8>u??=_?\>???=???>??U?u	>)$??????e?5<=?S޽??/?9`?=s??=a!?=F??3?<?kռ?#=n??==?=?w=?o?m?e2?UU=mmѼ??????=???='ý[@ǻc?a˼F??<v<=@?н???=?:?=?./?,y???
>?Qܺ+?<l?<_??@?+?? j???=?\?=?`/??y=?Ѳ???6=8b?8???"=M???{;>?w??C.!>R?7???A???????X?`>???=h?.???U=?疽Z?8=?"?=
Dؼ????5Y ?J???/[C=??,??~>|?=L?v?/=??!???V(=zM>??<?F??vC??`??i(>?<>???=?v==GܼA t?????ѓ	???=焽?F?m???<?T????;>b?p=!??lV??%	??F}>?q|=?|???C?;?	Խv	I??-?;dJ?@?[<wa?=?O{??,??G?&??h?=V.??"??A"Ƽ?d={??=??=_?[?7R??_P=?X:>?W=???<,z#?
?????	???w|??3Ҽ??<???=??<\e\?B??=H????w&=?r?=-`??	?<?Om????<?!?=r䡼?????2X<?<?F#???.>??;?.???Ƶ;h?=\>????˼t%>?w??Ǉ?ف??BI;)??r???? ???0a=??*???:?x?>>6?l<A?^=????'=J^?=zKt??? >,?=????-68?n?9???5/=?\?=?e?=6|????=?Ǽ???=?5C?1l?ϡK=U??=?Q>?͘=?0??]?=?R???>????$?=??a=Y=?=??<???=?T<?`?0:]=???eI;?qm???{=??ȼ?)?=????k??=??1=&???׼?a?=??%>оʼ?tq>N?=?#??V1?˹I=??&?н{????P'????=[?=?ʠ?D#?ִ?=?{?k#?=|??=5???Sf?((պ?oG??{?????K\???=??=j?=`?=.??=u??D	??Ĳ?=֢?Xܽ(m޽C??j?=L\?%os;5???ݼ?(ؼ?>?=,y*<, ">??=ϵ>?FսT?=?ؽ?>?%?=6}?<?*?+م??>????Jۻ??2=O?;???ټ?~ʼn??P?>{??????_=?K>A^?=??Ƚ???v??=@Mn<??ļ8??!VG???ʼ??<??:>?b?<?L??$?>5?b?G??=????sy!>i???\??;M:????=O?Ͻ ɽ??>3?e>}.F?2?>??t?i?=???=??м d?=?E>H
=I>/? >??I??N???ʽ??U=?N???l~??_*>?ޒ=v?D<#??@j >?/?C?Y<%p?<3???]=?Գ=?B>?,ƽ??>?U=??ԻK<V>Qas9+??=??<??=?!?V??<^??=?	M>4?a=g?????.??첻??-??=???<`?>ZFd=IƤ?W?#>t??;?&???I޼El?=ǣѽx(#>?߫?Ӈؽ?<?<???<?U???>?S3?7??<?»????$?b?b?!?????;????:?o??CΜ=??	=?i=$sw????=(?w???輅p???o??y_?=i?Q????<Q??<?U<?cG???>?>=?l>	1}?Њ=G???4?=?!=?(=ǌ??	K?=??????=;?н??0???"<??R=D{V<%??]??x??b?? ?=??>?l????=???>u?M=??.=?0=E%?=w?Z???>?d?$?_??%F?iϽ?н??_?v??=?2:=9z?<?SW=?&2????P<??~Ե<??=?:Ͻ^ʴ???<i.?????<`????O?=?????;?=?Z?=?1??*?>Ts/??k?<?>;?i???N?<=^S=???=?D?;-ף?Z????н????;IX??^}=p?H???!>c/.>?YP?g??ov?<?Q#>?߽?? =?<>hۄ=??^??sǽt?=?L>X??=An?ߐ???Y<ר?Bm=?=???=͞??ʮ|= %???r%=?>O?߽??{=}]@???????????8=I???rQ=????ݲ?=q??Me??0[??={逽](>=*?<??=??ڼyb?=?>꫽?ޥ????=7????^?>:??=6J?=/@??,׼D?I>aH=}??.f?=(v??a?>?2<???'?i<?M	>%??XS1=O΂=`???S???Fm<? >???=/??k?=??z=/?>J?< V?: ??<a6X>?d???? ??/<=y?ý?>*t?;?+???!=??'=???=n???;n=ذ&;?h˼X߽w?];??۽????ҹ?<?1=?3???9ϼr??7,>??i?_?9???>?>???=N?[=ieU????m+?=="?<wl??``?? >?ᮽ??????=ڑ>?rI??=w?????????<?/?:U`۽? ?=?ǲ?e#????=n???t?P?/=?u*?@?=?½с>??"v;??=???<޷?=N*ʽju??N????D??#=m??;?9"?]?ؼ???؎2>>?;cH[??aX???7??[?<&=ȯw??YF??vƽ???=ü6q`????=<?]?x'Q?C?#??.3??{<):1?N??=??7>s?s?Rc?#O?>???=VǼ16T??
?=?Ø???<????"]q??r}=?!m=M3?(??;??=st;Aj????6=??#??<?=?l?<m????G2?3????L>"?>??=?N???W>K?=?1?<???=????????>7ƽ?	???ҽH2??;?=??=H`0?=?&??4#=0??=3딽?>?*?=?0>\˴?y?L>;?.>??>?)?F?軁'???I???QR<????j??#?V>eD??'X??n?<??:??>=΋}=0?+??}>3???9?=??4=???x{->??1??g<x??~j?wם>??#>???k?@=;?=bM>?Գ<? ??Ɵ?ܵ???[9??;ٽi?)>??c>?e>FM=_Zͼ????)???f2=?껊踽>e????m?E=?oT=?|???=w?=??%??@{b??-?	f???#?=??Y?:wE?`Ps?6n:q.%??l@?RƼ?hP??<6>bd>(??P?N???	?ZJ???k!???%=???<??C>3??=???t??????:?????7??r?>???=u=?_0???޺=><{??Λ?=?a/>̕J=Yk=:)??T???|?=?<??)P??*x?=??=j	?3??<M?=chս@%>P??=??`>?PM=?&?;?ѣ=C???jBʽ6?">{\?<!?==????'?T?8?h??=?>?7m<;)?<^&??ߗ????< ?t???1?φ"<?a?????<>?<??<??>	?>??!?P?D?h?H>?.?<??o>?ν֟?=R????<?T?f?:&??R?>ä??{?:=?w???6?uj??cs?=d??=?>??R?Z??*?v??Z?=ʀU>?????4?=?#??????CN?@N(>i?EK?<????H??=8Ϻ jM?	Xݼָ??v??<nG>??S<? ?????=?b;=??=g?=??I>?0ͻ???=>i󽗚s=i/(?Txa<a?4?rr??n??pDO=??ν8K???>?Q?????9??????K?=??ӻmj+=UyԻ???=sGu?OfR?f?q=W?e<???=l???pb??'=>?-?=?࠽vZ?=??޽67>"\??0}h?Ѧƽ??G=Fi7???D?N?R????=?^?w??????=?͗??c<????U,[?^?=Q/?<?b.>}??;??<=????<????0??8????=?=?(?r&???:Q&????<?<>?[=SoؽYt???j_=G?n??G;ZvJ?A?=
????C??X?=?@?<??<%??=<??=h???(>?(%>?I??????ۼ|G?Qhӽi!"?	n?KB?=e⢽?G?=?sٽ)<??	<?5&?z?=??_k??˪<?׌??̽?E=???<???`Z=Ƞ?=??>bzG?4??=[<???=?\?=?p̻??D<`!<?"??Xo?=?ջ=]>?ٿ?-t?=?c???r	???
>S'??(????N<???U?ļ։?0???7??????e????<.??=?WC?dk?<?r=?? =nqy??f.>?t>?HV=?$?=?S7?(?4>?8?=@?X=4bd<z?0=?`??? ?<???̡?=?m??ºW???=?A.>(㉾>?????>Ϛ??67>(x:<??=] }<?/?=f??;?֘:?m???B<^	?? 
d??_??K??<?Z?<S[??(1???????=?????????<??>M?P>k1?=\????o=Py>??B?it?;j
??z?X@?=????H???J,=?X???=?j<?;?Ti?=?Pڼ??<lp?=??=˛??c?	=?-???5?*<oټ끮?????܎=Y??;l	=?@?=?v=?9R=?p??
>???;?b~=%@{???=????ܴ??$??=??d<??!=?A(<???=?????<??z?[???<O??w?<o+????#??6>????A??,??=??E???<mU??????|???N?_=?)?=??>o͛?(>??=?N)>y?P=$%6=j??=-?M>???????=??=?+-??>??^?+=f^Ѽ3??<??f?u5?[?=?u?=8?=??o?R???Z?5\?l?	>????Ҧ+????^	?	.=?8>T?????$?o/??ԗ>?d=? ?=?_?=?8?N???i=O ?;=?=??uɽ}??;?=｟p,<?=?Lo??!??w??=^p??6k???)?=;??<:^)>]_S??????=??:??????????z??C???B=????????=??˽E??;p??=??????ͽ??a>W6????i???PzB=3ք=Tu?=?m'>??w=?fؽ?C6???=? ?aQ?=??=??(=և?<^???{L??%?=???????????p??߳='????=???=?l??"??,?=?a?`ϋ?q?I?P?!?4??ˑ??r??4??T8??Ռ?7?_???0=?????bz<H{;>PTX?F?@???
<??潸?$???=?W'??|y?ѥ?=f?T???c=@???Vj>?,=??=??w>?>??u<???=???<oF\????;ǰ?????=9.?I<??zv?jKͽqW½t???l=Y?+8&?=????g???Ѽȫ.>??c=?֔???<~??kՒ?^??<?Y??	???f=w?=?b???\p=? <B+?=?)6=??콙??=??>??}:?!?w???"?~??C??ܶ??6s=????A?<í?=6?=x?">???????TT?=:&?<>????}??Bz?q???>F?M=dW?n?	>^??>% ????????k?$>W????n?=???<-
??P?(????=?*?=??=m???'????<N???)?<??>???=?ϧ=???<?Z>E????3O??^=
bi?N?X?!???¹??#j=? >? ,>b?=?#??????[???????=WX]= ?'?w?????kf>??<?O??=E????=&=?<?k?=?Z??i??=??ؼ???3E?&;???>?=?X?= ?M<?R?=?xD??Q??+>??=&?9>w?:???D?<+( ??????'??F?>???	=??ý4???v?<nq????=v÷=?t?B? ???𻲨 ??? =?C?=>>=??>xgԽ?Ķ=???=?+'?U?\?%Tｕ?/=?P??J?|>??=*
dtype0*
_output_shapes
:	? 
~
Variable
VariableV2*
shape:	? *
shared_name *
dtype0*
	container *
_output_shapes
:	? 
?
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
j
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
:	? 
?
Variable_1/initial_valueConst*?
value?B? "?=?>??Z<??V=???='|?=2L??T`Y<?;T=?7F>zp?=ӛ=??`>a???|??????>w???&s?:?+?ް?<?$?????=?@??????ֆI>Y??G!??R?x=!	?????=?~???6I?*
dtype0*
_output_shapes

: 
~

Variable_1
VariableV2*
shape
: *
shared_name *
dtype0*
	container *
_output_shapes

: 
?
Variable_1/AssignAssign
Variable_1Variable_1/initial_value*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
o
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes

: 
? 
Variable_2/initial_valueConst*? 
value? B?   "? \????=?l?>?V??i?;>)?=?l,? ?<?<L?)??C??R?d=]?>???1??=Y(M>?0??27?>???=????E(>?_?;??콻?z=RV??H?<?庽h??=A?V>?4)=?z???,?<??`???!>?T?;???<??=[jG?bZ????=،??;??? ???i?>%H??t??=?!?=???> -?>`?;>?PU>?????oY=ʻ>?&r???????#=߲ɽƃ??y??K>?b??~a??Y=?68<?􈽰4??????#??=?f??q?8[>?????:>$?d:???=???=ӊѼ??4=?(??s??g??ʠ???a<?? ???(??=???=???=[!c?iև>1??='???J????N>?_>T?????>??B??PԽ)?=???>Gm>?&<?U1?<rX=|?:???q>???>~F?<tx?=?M>w?K>?뀻l??=?#!?Bm?=??۽?D<?ü
?B>2??=?J?$?ێ???L?=g?4>?B>????ht=?k?????>?灾/D뽜;?=??+=u?+>?'??[?=?gM?⊴=?:
???j?]??<???0>Hq=?Y@=?a6??EQ?D?J<='?,=(~?=?R?=R.>?N>?	?>??;`jҽ??????뽗?R?Tɷ<?Es?s,=:`?<!x?;#?|>?+?ʳ4>l?|?	?ē߾ Q>?K??F?<???=?`?=?{{>GN$=?u???h>-Wm=???=BjB?T??=T??=Bߵ?L?Ҽ?s>K<?=??>????AZ?D??>??O>?Q ?x=\>!|?>)BX???ɽ??E?pr?=ǯ=??h???<?`C??R=??=E?????
?d?=J?c?}??=?*?>Hn?<??g=`j?????O+?????????'>ӌ??s?=7??>R?<???{???>=???b???\P?<3r??s==VZ?z???L???҇=?:Q???>?2ﾟi?>????S}a>^??<%???>	<R>???H? ?n????=?????????Խe<????L<E????????>??=???=?'??J>?-?>I??K??	)>??I??yR>?J??v???=??>#?̼Lp??????AG??L?>M??d]???Ё=???|/=C?e??k,>ҍӼa)=BӖ>b??=4撾??-?????5????g>|?Z>B?A>???<?'F??nU?d?׾b?н[E??&4??E>:???}?=r?j<>???w??_??=^??<8???P?Q<ؑ?_O???֋>??y?Y???ɾ??j??~??j:??(н?7??&i?>Ć??ȅ???C???ؽ?1?????<6aʽy3?=??>x??=^hl>? ͽv(??;?f>???=??;=?iE??詽?i>5fT?Ѿ???j?/|?=???<F?j?7R?<ɽ????=۸?9? <?????>?<???>*?"??3d?ؽ=??K=?G>?D<|?ǽ????X>P?j??Z?6X'>?J??|.?????D2??R?=?e??t4?=/W;2?#??J>~?H>? Q??_'??#ܾ?v??4=}??=-??<F?=wFk>YGJ?x$3>h??r???????v=P?q??`?:|:???oս?q????>π???6>?۹??j??6&t??N?
|??0??ն?>?.??/?P>+?@>??8?\OʽQRH????>?>>8,/??GK?k?>???Ĳ??2?W>??h?7hӾ???<?&???*???˟??R?>0?ƾ??!???=?Z?ߝ?e??=?pV>}?$=??>l?ǻ䴔={??=??(??^??̽K?>?R>?B?t>2??=b͝?_j??Y????n?=?>"sW=??=&????q???=??JN=?GT=?C?>?&=??<ħ?=?c?????=?EU=7Ul?\9>E?F>kj?>'?=vk"?*Ɂ?ы???J=>?ؼ=Qh???!?=? ?q)?S:3>?/???O?HK???jG>??:?o:f??ΰ???m?w??<=??????.??>?
ٽx????r)=??>f???Z?>???>RZ
>L??? >?sl>?? =??^?\??>?G?m<?????=,Q?=FB?=?>;+?>]?=͎?=?US?#N	?ĉ=w9??=??Ĵ?<?L?<5?|>?e??Me?L>HjL?DL.??????X=?0 ??Q???????t?J?.==i?<H3?<O??=?ͳ??E$??־??b???q>?B1>^??<N????6?j?6>r?g>=;????+???KN?=?+@??[]>n?>????F8A???i=(&<??Y????>?{??k >??=J~;???<4
?=?/?PX>?˞?)?7>h?/>"?>?Y??%n???6>;?>????X?<>ڸ?=??=6_3?Zr?;??@>A?=?W???Z??ߋ=??5>?;?L????]??쒽???=	!?=;k???нϩ???sy?bTQ=??0>??/???ν???=??F???a?t!?=?6>??k??P?<???ؤ????/?`????[?????=.k?=?,>0D?=????ܽr?Ὂ?????1????[?? @?9B?1??ƍ>R?????#?R6??????????I|<?????b?>?H?=?T??D<?#?<ˏ?>??"??ss>L]??%?=??1???;?ֻ>o??t???????̼??>?v@?9??????U??N>n4?=?R??&>H?4>l?>?ռ<?|>~\Ⱦ??????=?a?>?fŻ.???vI>V? =??r<<B?>Z?p>?G'????>???=	???[?r??m=wcҽD?=x??=7??gӎ>?e:???E^>_&>?E>??=1??????Y݊????=??=#?M>,5U??K>5t?=옂=?V0=???Ɩ۾?a???}>~>?̾?đ??]оk֖??F??????M??!1??՘?
@<??B??z>???=??G>???=?V?M?#ѿ?????z=??;??????G?=???=?|??+?E???ʋ?>???<a7/?̽Ƹl??'???~>?????"?> ?>}Qp>???=a?
>Y?y?>?!>??;??B>Q6=?Db?M1k??D??!w?>?qf???9>??	??|?= A>?l"????=QA?>? ɽ?p???=?mݽXG<?i?@>?㍾shi>s??>?s>????_???f	?!۴>?M?>]>}??>???=??>?z?=?Q??
??鱇=P?(?4?-?/?M????K?>҄?/{?? ??iw?=_?=?\C?&??Խ?!ֈ??d꼩???/?b=??"?Xܓ>?&/??7?=zAN?p?X>	~=?Ľ
?????O>}*<?m?>}Pؽ?F?n?=7?׽?ԉ=潹??Bw?W??=???%?n???Ž???=3p<?#?ݽe??Q%Ѿ??Ƚ:?ν@?????̾?l?3(?=]?0>?PM>?=???r?=ovʽyQm?3?F??-??@|A?F?Z??????^<=?#m=???????1???B<̼??Y?	:>?đ>B#2=s};?G==e	
>8??>L?*??m???}ֻs?=??w??rj<?8??ϥ=?%z>E?>???>??f=??1<??tX?>???=??f??Z^?D?x=?u?>i:?=??=99?????_?a>KN?=?Gg=@??`>?\??Bmj>]o?h?H?9jQ????=L??@e>?a??;6???Լx?d=??L>???<??h??M1=??=??Yk??W?=h??ڔ?t8=H?S<m???Qau>?o>}?????????</???_S????(>h=?=-_>9J?>?i&=??>ߑݽ׀???1>?S?b
X>?֐>??*???>>G??>????/?=??u?̛????;? ???Z>c???c?>h>[>??=&??NI=???k'?/?<a???~g.?)B
>	G??.>"???O:??L?֋$??V2???6>tU>??U=Q>F????q?сr>??=?ᬽ?a?(??>`1<=?*v>?2>???B???:?8?+????<?-?QVn=???;??>?P?>U??=-;>?????m>???:????"#X????=s??=3 =7?-?'O??T>W?G=??n?C?t>QvA>????뚾??;*
dtype0*
_output_shapes

:  
~

Variable_2
VariableV2*
shape
:  *
shared_name *
dtype0*
	container *
_output_shapes

:  
?
Variable_2/AssignAssign
Variable_2Variable_2/initial_value*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:  
?
Variable_3/initial_valueConst*?
value?B? "??k =[>?>?<??X*?<X$y?e-?=nM??$?=?m?????۝???ٽ?R?f??????Y>J???=\hV?\{'????=Jް>???=zuw?E6s???[=G?????Ƹ/????H+????Z?*
dtype0*
_output_shapes

: 
~

Variable_3
VariableV2*
shape
: *
shared_name *
dtype0*
	container *
_output_shapes

: 
?
Variable_3/AssignAssign
Variable_3Variable_3/initial_value*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
o
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes

: 
?
Variable_4/initial_valueConst*?
value?B?<"?<??;?~?:??̼>%?<??ǯ??Ѭ??Tؽ?lB=L??=?6?>^?5=?_?=M?w(??8t?>??4???=?m>Μ?=?l???;Tμ??Ľv#:??q>?? ?Zry??@G?n?P???P>@??=Q^7?:?Ľ?[=??i????? ?b>r·>??<???'X录?\>]i
??????A??%???ID??˽B?m??7>??<g\d???~?DO>?[Q?4}/>)?$>s??=K
K<*
dtype0*
_output_shapes

:<
~

Variable_4
VariableV2*
shape
:<*
shared_name *
dtype0*
	container *
_output_shapes

:<
?
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:<
J
ConstConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: 
n

Variable_5
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
?
Variable_5/AssignAssign
Variable_5Const*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
g
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
: 
g
predPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
&embedding_lookup/embedding_lookup/axisConst*%
_class
loc:@feature_embeddings*
value	B : *
dtype0*
_output_shapes
: 
?
!embedding_lookup/embedding_lookupGatherV2feature_embeddings/read
feat_index&embedding_lookup/embedding_lookup/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*%
_class
loc:@feature_embeddings*4
_output_shapes"
 :??????????????????
?
*embedding_lookup/embedding_lookup/IdentityIdentity!embedding_lookup/embedding_lookup*
T0*4
_output_shapes"
 :??????????????????
s
embedding_lookup/Reshape/shapeConst*!
valueB"????      *
dtype0*
_output_shapes
:
?
embedding_lookup/ReshapeReshape
feat_valueembedding_lookup/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
embedding_lookup/MulMul*embedding_lookup/embedding_lookup/Identityembedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
?
!first_order/embedding_lookup/axisConst*
_class
loc:@feature_bias*
value	B : *
dtype0*
_output_shapes
: 
?
first_order/embedding_lookupGatherV2feature_bias/read
feat_index!first_order/embedding_lookup/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_class
loc:@feature_bias*4
_output_shapes"
 :??????????????????
?
%first_order/embedding_lookup/IdentityIdentityfirst_order/embedding_lookup*
T0*4
_output_shapes"
 :??????????????????
?
first_order/MulMul%first_order/embedding_lookup/Identityembedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
c
!first_order/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
first_order/SumSumfirst_order/Mul!first_order/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*'
_output_shapes
:?????????
i
first_order/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!first_order/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!first_order/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
first_order/strided_sliceStridedSlicedropout_keep_fmfirst_order/strided_slice/stack!first_order/strided_slice/stack_1!first_order/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
first_order/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
e
first_order/subSubfirst_order/sub/xfirst_order/strided_slice*
T0*
_output_shapes
: 
h
first_order/dropout/ShapeShapefirst_order/Sum*
T0*
out_type0*
_output_shapes
:
k
&first_order/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&first_order/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
0first_order/dropout/random_uniform/RandomUniformRandomUniformfirst_order/dropout/Shape*
seed?*
T0*
dtype0*
seed2G*'
_output_shapes
:?????????
?
&first_order/dropout/random_uniform/subSub&first_order/dropout/random_uniform/max&first_order/dropout/random_uniform/min*
T0*
_output_shapes
: 
?
&first_order/dropout/random_uniform/mulMul0first_order/dropout/random_uniform/RandomUniform&first_order/dropout/random_uniform/sub*
T0*'
_output_shapes
:?????????
?
"first_order/dropout/random_uniformAdd&first_order/dropout/random_uniform/mul&first_order/dropout/random_uniform/min*
T0*'
_output_shapes
:?????????
^
first_order/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
k
first_order/dropout/subSubfirst_order/dropout/sub/xfirst_order/sub*
T0*
_output_shapes
: 
b
first_order/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 

first_order/dropout/truedivRealDivfirst_order/dropout/truediv/xfirst_order/dropout/sub*
T0*
_output_shapes
: 
?
 first_order/dropout/GreaterEqualGreaterEqual"first_order/dropout/random_uniformfirst_order/sub*
T0*'
_output_shapes
:?????????
~
first_order/dropout/mulMulfirst_order/Sumfirst_order/dropout/truediv*
T0*'
_output_shapes
:?????????
?
first_order/dropout/CastCast first_order/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*'
_output_shapes
:?????????
?
first_order/dropout/mul_1Mulfirst_order/dropout/mulfirst_order/dropout/Cast*
T0*'
_output_shapes
:?????????
d
"second_order/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
second_order/SumSumembedding_lookup/Mul"second_order/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*'
_output_shapes
:?????????
a
second_order/SquareSquaresecond_order/Sum*
T0*'
_output_shapes
:?????????
\
SquareSquareembedding_lookup/Mul*
T0*+
_output_shapes
:?????????
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
x
SumSumSquareSum/reduction_indices*

Tidx0*
	keep_dims( *
T0*'
_output_shapes
:?????????
V
SubSubsecond_order/SquareSum*
T0*'
_output_shapes
:?????????
J
mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
H
mulMulmul/xSub*
T0*'
_output_shapes
:?????????
]
strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
strided_sliceStridedSlicedropout_keep_fmstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
L
sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
E
sub_1Subsub_1/xstrided_slice*
T0*
_output_shapes
: 
P
dropout/ShapeShapemul*
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
 *  ??*
dtype0*
_output_shapes
: 
?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
seed?*
T0*
dtype0*
seed2e*'
_output_shapes
:?????????
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
?
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*'
_output_shapes
:?????????
?
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*'
_output_shapes
:?????????
R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
I
dropout/subSubdropout/sub/xsub_1*
T0*
_output_shapes
: 
V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
[
dropout/truedivRealDivdropout/truediv/xdropout/sub*
T0*
_output_shapes
: 
u
dropout/GreaterEqualGreaterEqualdropout/random_uniformsub_1*
T0*'
_output_shapes
:?????????
Z
dropout/mulMulmuldropout/truediv*
T0*'
_output_shapes
:?????????
{
dropout/CastCastdropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*'
_output_shapes
:?????????
a
dropout/mul_1Muldropout/muldropout/Cast*
T0*'
_output_shapes
:?????????
m
deep_component/Reshape/shapeConst*
valueB"?????   *
dtype0*
_output_shapes
:
?
deep_component/ReshapeReshapeembedding_lookup/Muldeep_component/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:??????????
l
"deep_component/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$deep_component/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
n
$deep_component/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
deep_component/strided_sliceStridedSlicedropout_keep_deep"deep_component/strided_slice/stack$deep_component/strided_slice/stack_1$deep_component/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Y
deep_component/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
n
deep_component/subSubdeep_component/sub/xdeep_component/strided_slice*
T0*
_output_shapes
: 
r
deep_component/dropout/ShapeShapedeep_component/Reshape*
T0*
out_type0*
_output_shapes
:
n
)deep_component/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
)deep_component/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
3deep_component/dropout/random_uniform/RandomUniformRandomUniformdeep_component/dropout/Shape*
seed?*
T0*
dtype0*
seed2|*(
_output_shapes
:??????????
?
)deep_component/dropout/random_uniform/subSub)deep_component/dropout/random_uniform/max)deep_component/dropout/random_uniform/min*
T0*
_output_shapes
: 
?
)deep_component/dropout/random_uniform/mulMul3deep_component/dropout/random_uniform/RandomUniform)deep_component/dropout/random_uniform/sub*
T0*(
_output_shapes
:??????????
?
%deep_component/dropout/random_uniformAdd)deep_component/dropout/random_uniform/mul)deep_component/dropout/random_uniform/min*
T0*(
_output_shapes
:??????????
a
deep_component/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
t
deep_component/dropout/subSubdeep_component/dropout/sub/xdeep_component/sub*
T0*
_output_shapes
: 
e
 deep_component/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
deep_component/dropout/truedivRealDiv deep_component/dropout/truediv/xdeep_component/dropout/sub*
T0*
_output_shapes
: 
?
#deep_component/dropout/GreaterEqualGreaterEqual%deep_component/dropout/random_uniformdeep_component/sub*
T0*(
_output_shapes
:??????????
?
deep_component/dropout/mulMuldeep_component/Reshapedeep_component/dropout/truediv*
T0*(
_output_shapes
:??????????
?
deep_component/dropout/CastCast#deep_component/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*(
_output_shapes
:??????????
?
deep_component/dropout/mul_1Muldeep_component/dropout/muldeep_component/dropout/Cast*
T0*(
_output_shapes
:??????????
?
deep_component/MatMulMatMuldeep_component/dropout/mul_1Variable/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
s
deep_component/AddAdddeep_component/MatMulVariable_1/read*
T0*'
_output_shapes
:????????? 
a
deep_component/ReluReludeep_component/Add*
T0*'
_output_shapes
:????????? 
n
$deep_component/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&deep_component/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&deep_component/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
deep_component/strided_slice_1StridedSlicedropout_keep_deep$deep_component/strided_slice_1/stack&deep_component/strided_slice_1/stack_1&deep_component/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
[
deep_component/sub_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
t
deep_component/sub_1Subdeep_component/sub_1/xdeep_component/strided_slice_1*
T0*
_output_shapes
: 
q
deep_component/dropout_1/ShapeShapedeep_component/Relu*
T0*
out_type0*
_output_shapes
:
p
+deep_component/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
+deep_component/dropout_1/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
5deep_component/dropout_1/random_uniform/RandomUniformRandomUniformdeep_component/dropout_1/Shape*
seed?*
T0*
dtype0*
seed2?*'
_output_shapes
:????????? 
?
+deep_component/dropout_1/random_uniform/subSub+deep_component/dropout_1/random_uniform/max+deep_component/dropout_1/random_uniform/min*
T0*
_output_shapes
: 
?
+deep_component/dropout_1/random_uniform/mulMul5deep_component/dropout_1/random_uniform/RandomUniform+deep_component/dropout_1/random_uniform/sub*
T0*'
_output_shapes
:????????? 
?
'deep_component/dropout_1/random_uniformAdd+deep_component/dropout_1/random_uniform/mul+deep_component/dropout_1/random_uniform/min*
T0*'
_output_shapes
:????????? 
c
deep_component/dropout_1/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
z
deep_component/dropout_1/subSubdeep_component/dropout_1/sub/xdeep_component/sub_1*
T0*
_output_shapes
: 
g
"deep_component/dropout_1/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 deep_component/dropout_1/truedivRealDiv"deep_component/dropout_1/truediv/xdeep_component/dropout_1/sub*
T0*
_output_shapes
: 
?
%deep_component/dropout_1/GreaterEqualGreaterEqual'deep_component/dropout_1/random_uniformdeep_component/sub_1*
T0*'
_output_shapes
:????????? 
?
deep_component/dropout_1/mulMuldeep_component/Relu deep_component/dropout_1/truediv*
T0*'
_output_shapes
:????????? 
?
deep_component/dropout_1/CastCast%deep_component/dropout_1/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*'
_output_shapes
:????????? 
?
deep_component/dropout_1/mul_1Muldeep_component/dropout_1/muldeep_component/dropout_1/Cast*
T0*'
_output_shapes
:????????? 
?
deep_component/MatMul_1MatMuldeep_component/dropout_1/mul_1Variable_2/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:????????? 
w
deep_component/Add_1Adddeep_component/MatMul_1Variable_3/read*
T0*'
_output_shapes
:????????? 
e
deep_component/Relu_1Reludeep_component/Add_1*
T0*'
_output_shapes
:????????? 
n
$deep_component/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&deep_component/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&deep_component/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
deep_component/strided_slice_2StridedSlicedropout_keep_deep$deep_component/strided_slice_2/stack&deep_component/strided_slice_2/stack_1&deep_component/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
[
deep_component/sub_2/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
t
deep_component/sub_2Subdeep_component/sub_2/xdeep_component/strided_slice_2*
T0*
_output_shapes
: 
s
deep_component/dropout_2/ShapeShapedeep_component/Relu_1*
T0*
out_type0*
_output_shapes
:
p
+deep_component/dropout_2/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
+deep_component/dropout_2/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
5deep_component/dropout_2/random_uniform/RandomUniformRandomUniformdeep_component/dropout_2/Shape*
seed?*
T0*
dtype0*
seed2?*'
_output_shapes
:????????? 
?
+deep_component/dropout_2/random_uniform/subSub+deep_component/dropout_2/random_uniform/max+deep_component/dropout_2/random_uniform/min*
T0*
_output_shapes
: 
?
+deep_component/dropout_2/random_uniform/mulMul5deep_component/dropout_2/random_uniform/RandomUniform+deep_component/dropout_2/random_uniform/sub*
T0*'
_output_shapes
:????????? 
?
'deep_component/dropout_2/random_uniformAdd+deep_component/dropout_2/random_uniform/mul+deep_component/dropout_2/random_uniform/min*
T0*'
_output_shapes
:????????? 
c
deep_component/dropout_2/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
z
deep_component/dropout_2/subSubdeep_component/dropout_2/sub/xdeep_component/sub_2*
T0*
_output_shapes
: 
g
"deep_component/dropout_2/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 deep_component/dropout_2/truedivRealDiv"deep_component/dropout_2/truediv/xdeep_component/dropout_2/sub*
T0*
_output_shapes
: 
?
%deep_component/dropout_2/GreaterEqualGreaterEqual'deep_component/dropout_2/random_uniformdeep_component/sub_2*
T0*'
_output_shapes
:????????? 
?
deep_component/dropout_2/mulMuldeep_component/Relu_1 deep_component/dropout_2/truediv*
T0*'
_output_shapes
:????????? 
?
deep_component/dropout_2/CastCast%deep_component/dropout_2/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*'
_output_shapes
:????????? 
?
deep_component/dropout_2/mul_1Muldeep_component/dropout_2/muldeep_component/dropout_2/Cast*
T0*'
_output_shapes
:????????? 
U
deep_fm/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
deep_fm/concatConcatV2first_order/dropout/mul_1dropout/mul_1deep_component/dropout_2/mul_1deep_fm/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:?????????<
?

output/MulMatMuldeep_fm/concatVariable_4/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:?????????
`

output/AddAdd
output/MulVariable_5/read*
T0*'
_output_shapes
:?????????
T
sigmoid_outSigmoid
output/Add*
T0*'
_output_shapes
:?????????
X
loss/log_loss/add/yConst*
valueB
 *???3*
dtype0*
_output_shapes
: 
l
loss/log_loss/addAddsigmoid_outloss/log_loss/add/y*
T0*'
_output_shapes
:?????????
]
loss/log_loss/LogLogloss/log_loss/add*
T0*'
_output_shapes
:?????????
d
loss/log_loss/MulMullabelloss/log_loss/Log*
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
f
loss/log_loss/subSubloss/log_loss/sub/xlabel*
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
p
loss/log_loss/sub_1Subloss/log_loss/sub_1/xsigmoid_out*
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
loss/log_loss/Cast/xConst?^loss/log_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
w
loss/log_loss/Mul_2Mulloss/log_loss/sub_2loss/log_loss/Cast/x*
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
loss/log_loss/num_present/EqualEqualloss/log_loss/Cast/x!loss/log_loss/num_present/Equal/y*
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
[
global_step/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
o
global_step
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
?
global_step/AssignAssignglobal_stepglobal_step/initial_value*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step*
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
-gradients/loss/log_loss/value_grad/div_no_nanDivNoNangradients/Fillloss/log_loss/num_present*
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
&gradients/loss/log_loss/value_grad/mulMulgradients/Fill/gradients/loss/log_loss/value_grad/div_no_nan_2*
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
&gradients/loss/log_loss/Mul_2_grad/MulMul%gradients/loss/log_loss/Sum_grad/Tileloss/log_loss/Cast/x*
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
k
&gradients/loss/log_loss/Mul_grad/ShapeShapelabel*
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
&gradients/loss/log_loss/Mul_grad/Mul_1Mullabel$gradients/loss/log_loss/Neg_grad/Neg*
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
?
-gradients/loss/log_loss/Log_1_grad/Reciprocal
Reciprocalloss/log_loss/add_1>^gradients/loss/log_loss/Mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/Log_1_grad/mulMul=gradients/loss/log_loss/Mul_1_grad/tuple/control_dependency_1-gradients/loss/log_loss/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
+gradients/loss/log_loss/Log_grad/Reciprocal
Reciprocalloss/log_loss/add<^gradients/loss/log_loss/Mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
$gradients/loss/log_loss/Log_grad/mulMul;gradients/loss/log_loss/Mul_grad/tuple/control_dependency_1+gradients/loss/log_loss/Log_grad/Reciprocal*
T0*'
_output_shapes
:?????????
{
(gradients/loss/log_loss/add_1_grad/ShapeShapeloss/log_loss/sub_1*
T0*
out_type0*
_output_shapes
:
m
*gradients/loss/log_loss/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
8gradients/loss/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/add_1_grad/Shape*gradients/loss/log_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss/add_1_grad/SumSum&gradients/loss/log_loss/Log_1_grad/mul8gradients/loss/log_loss/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/add_1_grad/ReshapeReshape&gradients/loss/log_loss/add_1_grad/Sum(gradients/loss/log_loss/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients/loss/log_loss/add_1_grad/Sum_1Sum&gradients/loss/log_loss/Log_1_grad/mul:gradients/loss/log_loss/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients/loss/log_loss/add_1_grad/Reshape_1Reshape(gradients/loss/log_loss/add_1_grad/Sum_1*gradients/loss/log_loss/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
3gradients/loss/log_loss/add_1_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/add_1_grad/Reshape-^gradients/loss/log_loss/add_1_grad/Reshape_1
?
;gradients/loss/log_loss/add_1_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/add_1_grad/Reshape4^gradients/loss/log_loss/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/add_1_grad/Reshape*'
_output_shapes
:?????????
?
=gradients/loss/log_loss/add_1_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/add_1_grad/Reshape_14^gradients/loss/log_loss/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/add_1_grad/Reshape_1*
_output_shapes
: 
q
&gradients/loss/log_loss/add_grad/ShapeShapesigmoid_out*
T0*
out_type0*
_output_shapes
:
k
(gradients/loss/log_loss/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
6gradients/loss/log_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/loss/log_loss/add_grad/Shape(gradients/loss/log_loss/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients/loss/log_loss/add_grad/SumSum$gradients/loss/log_loss/Log_grad/mul6gradients/loss/log_loss/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients/loss/log_loss/add_grad/ReshapeReshape$gradients/loss/log_loss/add_grad/Sum&gradients/loss/log_loss/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
&gradients/loss/log_loss/add_grad/Sum_1Sum$gradients/loss/log_loss/Log_grad/mul8gradients/loss/log_loss/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/add_grad/Reshape_1Reshape&gradients/loss/log_loss/add_grad/Sum_1(gradients/loss/log_loss/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
1gradients/loss/log_loss/add_grad/tuple/group_depsNoOp)^gradients/loss/log_loss/add_grad/Reshape+^gradients/loss/log_loss/add_grad/Reshape_1
?
9gradients/loss/log_loss/add_grad/tuple/control_dependencyIdentity(gradients/loss/log_loss/add_grad/Reshape2^gradients/loss/log_loss/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/loss/log_loss/add_grad/Reshape*'
_output_shapes
:?????????
?
;gradients/loss/log_loss/add_grad/tuple/control_dependency_1Identity*gradients/loss/log_loss/add_grad/Reshape_12^gradients/loss/log_loss/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/add_grad/Reshape_1*
_output_shapes
: 
k
(gradients/loss/log_loss/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
u
*gradients/loss/log_loss/sub_1_grad/Shape_1Shapesigmoid_out*
T0*
out_type0*
_output_shapes
:
?
8gradients/loss/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/loss/log_loss/sub_1_grad/Shape*gradients/loss/log_loss/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients/loss/log_loss/sub_1_grad/SumSum;gradients/loss/log_loss/add_1_grad/tuple/control_dependency8gradients/loss/log_loss/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients/loss/log_loss/sub_1_grad/ReshapeReshape&gradients/loss/log_loss/sub_1_grad/Sum(gradients/loss/log_loss/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
(gradients/loss/log_loss/sub_1_grad/Sum_1Sum;gradients/loss/log_loss/add_1_grad/tuple/control_dependency:gradients/loss/log_loss/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
z
&gradients/loss/log_loss/sub_1_grad/NegNeg(gradients/loss/log_loss/sub_1_grad/Sum_1*
T0*
_output_shapes
:
?
,gradients/loss/log_loss/sub_1_grad/Reshape_1Reshape&gradients/loss/log_loss/sub_1_grad/Neg*gradients/loss/log_loss/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients/loss/log_loss/sub_1_grad/tuple/group_depsNoOp+^gradients/loss/log_loss/sub_1_grad/Reshape-^gradients/loss/log_loss/sub_1_grad/Reshape_1
?
;gradients/loss/log_loss/sub_1_grad/tuple/control_dependencyIdentity*gradients/loss/log_loss/sub_1_grad/Reshape4^gradients/loss/log_loss/sub_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/loss/log_loss/sub_1_grad/Reshape*
_output_shapes
: 
?
=gradients/loss/log_loss/sub_1_grad/tuple/control_dependency_1Identity,gradients/loss/log_loss/sub_1_grad/Reshape_14^gradients/loss/log_loss/sub_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/loss/log_loss/sub_1_grad/Reshape_1*'
_output_shapes
:?????????
?
gradients/AddNAddN9gradients/loss/log_loss/add_grad/tuple/control_dependency=gradients/loss/log_loss/sub_1_grad/tuple/control_dependency_1*
T0*;
_class1
/-loc:@gradients/loss/log_loss/add_grad/Reshape*
N*'
_output_shapes
:?????????
?
&gradients/sigmoid_out_grad/SigmoidGradSigmoidGradsigmoid_outgradients/AddN*
T0*'
_output_shapes
:?????????
i
gradients/output/Add_grad/ShapeShape
output/Mul*
T0*
out_type0*
_output_shapes
:
d
!gradients/output/Add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
/gradients/output/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/output/Add_grad/Shape!gradients/output/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/output/Add_grad/SumSum&gradients/sigmoid_out_grad/SigmoidGrad/gradients/output/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
!gradients/output/Add_grad/ReshapeReshapegradients/output/Add_grad/Sumgradients/output/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/output/Add_grad/Sum_1Sum&gradients/sigmoid_out_grad/SigmoidGrad1gradients/output/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
#gradients/output/Add_grad/Reshape_1Reshapegradients/output/Add_grad/Sum_1!gradients/output/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/output/Add_grad/tuple/group_depsNoOp"^gradients/output/Add_grad/Reshape$^gradients/output/Add_grad/Reshape_1
?
2gradients/output/Add_grad/tuple/control_dependencyIdentity!gradients/output/Add_grad/Reshape+^gradients/output/Add_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/output/Add_grad/Reshape*'
_output_shapes
:?????????
?
4gradients/output/Add_grad/tuple/control_dependency_1Identity#gradients/output/Add_grad/Reshape_1+^gradients/output/Add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/output/Add_grad/Reshape_1*
_output_shapes
: 
?
 gradients/output/Mul_grad/MatMulMatMul2gradients/output/Add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????<
?
"gradients/output/Mul_grad/MatMul_1MatMuldeep_fm/concat2gradients/output/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:<
z
*gradients/output/Mul_grad/tuple/group_depsNoOp!^gradients/output/Mul_grad/MatMul#^gradients/output/Mul_grad/MatMul_1
?
2gradients/output/Mul_grad/tuple/control_dependencyIdentity gradients/output/Mul_grad/MatMul+^gradients/output/Mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/output/Mul_grad/MatMul*'
_output_shapes
:?????????<
?
4gradients/output/Mul_grad/tuple/control_dependency_1Identity"gradients/output/Mul_grad/MatMul_1+^gradients/output/Mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/output/Mul_grad/MatMul_1*
_output_shapes

:<
d
"gradients/deep_fm/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
!gradients/deep_fm/concat_grad/modFloorModdeep_fm/concat/axis"gradients/deep_fm/concat_grad/Rank*
T0*
_output_shapes
: 
|
#gradients/deep_fm/concat_grad/ShapeShapefirst_order/dropout/mul_1*
T0*
out_type0*
_output_shapes
:
?
$gradients/deep_fm/concat_grad/ShapeNShapeNfirst_order/dropout/mul_1dropout/mul_1deep_component/dropout_2/mul_1*
T0*
out_type0*
N*&
_output_shapes
:::
?
*gradients/deep_fm/concat_grad/ConcatOffsetConcatOffset!gradients/deep_fm/concat_grad/mod$gradients/deep_fm/concat_grad/ShapeN&gradients/deep_fm/concat_grad/ShapeN:1&gradients/deep_fm/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
?
#gradients/deep_fm/concat_grad/SliceSlice2gradients/output/Mul_grad/tuple/control_dependency*gradients/deep_fm/concat_grad/ConcatOffset$gradients/deep_fm/concat_grad/ShapeN*
T0*
Index0*'
_output_shapes
:?????????
?
%gradients/deep_fm/concat_grad/Slice_1Slice2gradients/output/Mul_grad/tuple/control_dependency,gradients/deep_fm/concat_grad/ConcatOffset:1&gradients/deep_fm/concat_grad/ShapeN:1*
T0*
Index0*'
_output_shapes
:?????????
?
%gradients/deep_fm/concat_grad/Slice_2Slice2gradients/output/Mul_grad/tuple/control_dependency,gradients/deep_fm/concat_grad/ConcatOffset:2&gradients/deep_fm/concat_grad/ShapeN:2*
T0*
Index0*'
_output_shapes
:????????? 
?
.gradients/deep_fm/concat_grad/tuple/group_depsNoOp$^gradients/deep_fm/concat_grad/Slice&^gradients/deep_fm/concat_grad/Slice_1&^gradients/deep_fm/concat_grad/Slice_2
?
6gradients/deep_fm/concat_grad/tuple/control_dependencyIdentity#gradients/deep_fm/concat_grad/Slice/^gradients/deep_fm/concat_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/deep_fm/concat_grad/Slice*'
_output_shapes
:?????????
?
8gradients/deep_fm/concat_grad/tuple/control_dependency_1Identity%gradients/deep_fm/concat_grad/Slice_1/^gradients/deep_fm/concat_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/deep_fm/concat_grad/Slice_1*'
_output_shapes
:?????????
?
8gradients/deep_fm/concat_grad/tuple/control_dependency_2Identity%gradients/deep_fm/concat_grad/Slice_2/^gradients/deep_fm/concat_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/deep_fm/concat_grad/Slice_2*'
_output_shapes
:????????? 
?
.gradients/first_order/dropout/mul_1_grad/ShapeShapefirst_order/dropout/mul*
T0*
out_type0*
_output_shapes
:
?
0gradients/first_order/dropout/mul_1_grad/Shape_1Shapefirst_order/dropout/Cast*
T0*
out_type0*
_output_shapes
:
?
>gradients/first_order/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/first_order/dropout/mul_1_grad/Shape0gradients/first_order/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
,gradients/first_order/dropout/mul_1_grad/MulMul6gradients/deep_fm/concat_grad/tuple/control_dependencyfirst_order/dropout/Cast*
T0*'
_output_shapes
:?????????
?
,gradients/first_order/dropout/mul_1_grad/SumSum,gradients/first_order/dropout/mul_1_grad/Mul>gradients/first_order/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
0gradients/first_order/dropout/mul_1_grad/ReshapeReshape,gradients/first_order/dropout/mul_1_grad/Sum.gradients/first_order/dropout/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
.gradients/first_order/dropout/mul_1_grad/Mul_1Mulfirst_order/dropout/mul6gradients/deep_fm/concat_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
.gradients/first_order/dropout/mul_1_grad/Sum_1Sum.gradients/first_order/dropout/mul_1_grad/Mul_1@gradients/first_order/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
2gradients/first_order/dropout/mul_1_grad/Reshape_1Reshape.gradients/first_order/dropout/mul_1_grad/Sum_10gradients/first_order/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
9gradients/first_order/dropout/mul_1_grad/tuple/group_depsNoOp1^gradients/first_order/dropout/mul_1_grad/Reshape3^gradients/first_order/dropout/mul_1_grad/Reshape_1
?
Agradients/first_order/dropout/mul_1_grad/tuple/control_dependencyIdentity0gradients/first_order/dropout/mul_1_grad/Reshape:^gradients/first_order/dropout/mul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/first_order/dropout/mul_1_grad/Reshape*'
_output_shapes
:?????????
?
Cgradients/first_order/dropout/mul_1_grad/tuple/control_dependency_1Identity2gradients/first_order/dropout/mul_1_grad/Reshape_1:^gradients/first_order/dropout/mul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/first_order/dropout/mul_1_grad/Reshape_1*'
_output_shapes
:?????????
m
"gradients/dropout/mul_1_grad/ShapeShapedropout/mul*
T0*
out_type0*
_output_shapes
:
p
$gradients/dropout/mul_1_grad/Shape_1Shapedropout/Cast*
T0*
out_type0*
_output_shapes
:
?
2gradients/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout/mul_1_grad/Shape$gradients/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
 gradients/dropout/mul_1_grad/MulMul8gradients/deep_fm/concat_grad/tuple/control_dependency_1dropout/Cast*
T0*'
_output_shapes
:?????????
?
 gradients/dropout/mul_1_grad/SumSum gradients/dropout/mul_1_grad/Mul2gradients/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
$gradients/dropout/mul_1_grad/ReshapeReshape gradients/dropout/mul_1_grad/Sum"gradients/dropout/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
"gradients/dropout/mul_1_grad/Mul_1Muldropout/mul8gradients/deep_fm/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
"gradients/dropout/mul_1_grad/Sum_1Sum"gradients/dropout/mul_1_grad/Mul_14gradients/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
&gradients/dropout/mul_1_grad/Reshape_1Reshape"gradients/dropout/mul_1_grad/Sum_1$gradients/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
-gradients/dropout/mul_1_grad/tuple/group_depsNoOp%^gradients/dropout/mul_1_grad/Reshape'^gradients/dropout/mul_1_grad/Reshape_1
?
5gradients/dropout/mul_1_grad/tuple/control_dependencyIdentity$gradients/dropout/mul_1_grad/Reshape.^gradients/dropout/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_1_grad/Reshape*'
_output_shapes
:?????????
?
7gradients/dropout/mul_1_grad/tuple/control_dependency_1Identity&gradients/dropout/mul_1_grad/Reshape_1.^gradients/dropout/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dropout/mul_1_grad/Reshape_1*'
_output_shapes
:?????????
?
3gradients/deep_component/dropout_2/mul_1_grad/ShapeShapedeep_component/dropout_2/mul*
T0*
out_type0*
_output_shapes
:
?
5gradients/deep_component/dropout_2/mul_1_grad/Shape_1Shapedeep_component/dropout_2/Cast*
T0*
out_type0*
_output_shapes
:
?
Cgradients/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/deep_component/dropout_2/mul_1_grad/Shape5gradients/deep_component/dropout_2/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients/deep_component/dropout_2/mul_1_grad/MulMul8gradients/deep_fm/concat_grad/tuple/control_dependency_2deep_component/dropout_2/Cast*
T0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_2/mul_1_grad/SumSum1gradients/deep_component/dropout_2/mul_1_grad/MulCgradients/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients/deep_component/dropout_2/mul_1_grad/ReshapeReshape1gradients/deep_component/dropout_2/mul_1_grad/Sum3gradients/deep_component/dropout_2/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
3gradients/deep_component/dropout_2/mul_1_grad/Mul_1Muldeep_component/dropout_2/mul8gradients/deep_fm/concat_grad/tuple/control_dependency_2*
T0*'
_output_shapes
:????????? 
?
3gradients/deep_component/dropout_2/mul_1_grad/Sum_1Sum3gradients/deep_component/dropout_2/mul_1_grad/Mul_1Egradients/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients/deep_component/dropout_2/mul_1_grad/Reshape_1Reshape3gradients/deep_component/dropout_2/mul_1_grad/Sum_15gradients/deep_component/dropout_2/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:????????? 
?
>gradients/deep_component/dropout_2/mul_1_grad/tuple/group_depsNoOp6^gradients/deep_component/dropout_2/mul_1_grad/Reshape8^gradients/deep_component/dropout_2/mul_1_grad/Reshape_1
?
Fgradients/deep_component/dropout_2/mul_1_grad/tuple/control_dependencyIdentity5gradients/deep_component/dropout_2/mul_1_grad/Reshape?^gradients/deep_component/dropout_2/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/deep_component/dropout_2/mul_1_grad/Reshape*'
_output_shapes
:????????? 
?
Hgradients/deep_component/dropout_2/mul_1_grad/tuple/control_dependency_1Identity7gradients/deep_component/dropout_2/mul_1_grad/Reshape_1?^gradients/deep_component/dropout_2/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/deep_component/dropout_2/mul_1_grad/Reshape_1*'
_output_shapes
:????????? 
{
,gradients/first_order/dropout/mul_grad/ShapeShapefirst_order/Sum*
T0*
out_type0*
_output_shapes
:
q
.gradients/first_order/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
<gradients/first_order/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/first_order/dropout/mul_grad/Shape.gradients/first_order/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
*gradients/first_order/dropout/mul_grad/MulMulAgradients/first_order/dropout/mul_1_grad/tuple/control_dependencyfirst_order/dropout/truediv*
T0*'
_output_shapes
:?????????
?
*gradients/first_order/dropout/mul_grad/SumSum*gradients/first_order/dropout/mul_grad/Mul<gradients/first_order/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients/first_order/dropout/mul_grad/ReshapeReshape*gradients/first_order/dropout/mul_grad/Sum,gradients/first_order/dropout/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
,gradients/first_order/dropout/mul_grad/Mul_1Mulfirst_order/SumAgradients/first_order/dropout/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
,gradients/first_order/dropout/mul_grad/Sum_1Sum,gradients/first_order/dropout/mul_grad/Mul_1>gradients/first_order/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
0gradients/first_order/dropout/mul_grad/Reshape_1Reshape,gradients/first_order/dropout/mul_grad/Sum_1.gradients/first_order/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
7gradients/first_order/dropout/mul_grad/tuple/group_depsNoOp/^gradients/first_order/dropout/mul_grad/Reshape1^gradients/first_order/dropout/mul_grad/Reshape_1
?
?gradients/first_order/dropout/mul_grad/tuple/control_dependencyIdentity.gradients/first_order/dropout/mul_grad/Reshape8^gradients/first_order/dropout/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/first_order/dropout/mul_grad/Reshape*'
_output_shapes
:?????????
?
Agradients/first_order/dropout/mul_grad/tuple/control_dependency_1Identity0gradients/first_order/dropout/mul_grad/Reshape_18^gradients/first_order/dropout/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/first_order/dropout/mul_grad/Reshape_1*
_output_shapes
: 
c
 gradients/dropout/mul_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
e
"gradients/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/dropout/mul_grad/MulMul5gradients/dropout/mul_1_grad/tuple/control_dependencydropout/truediv*
T0*'
_output_shapes
:?????????
?
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/Mul0gradients/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
 gradients/dropout/mul_grad/Mul_1Mulmul5gradients/dropout/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/Mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
?
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape*'
_output_shapes
:?????????
?
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1*
_output_shapes
: 
?
1gradients/deep_component/dropout_2/mul_grad/ShapeShapedeep_component/Relu_1*
T0*
out_type0*
_output_shapes
:
v
3gradients/deep_component/dropout_2/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Agradients/deep_component/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/deep_component/dropout_2/mul_grad/Shape3gradients/deep_component/dropout_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients/deep_component/dropout_2/mul_grad/MulMulFgradients/deep_component/dropout_2/mul_1_grad/tuple/control_dependency deep_component/dropout_2/truediv*
T0*'
_output_shapes
:????????? 
?
/gradients/deep_component/dropout_2/mul_grad/SumSum/gradients/deep_component/dropout_2/mul_grad/MulAgradients/deep_component/dropout_2/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
3gradients/deep_component/dropout_2/mul_grad/ReshapeReshape/gradients/deep_component/dropout_2/mul_grad/Sum1gradients/deep_component/dropout_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_2/mul_grad/Mul_1Muldeep_component/Relu_1Fgradients/deep_component/dropout_2/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_2/mul_grad/Sum_1Sum1gradients/deep_component/dropout_2/mul_grad/Mul_1Cgradients/deep_component/dropout_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients/deep_component/dropout_2/mul_grad/Reshape_1Reshape1gradients/deep_component/dropout_2/mul_grad/Sum_13gradients/deep_component/dropout_2/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
<gradients/deep_component/dropout_2/mul_grad/tuple/group_depsNoOp4^gradients/deep_component/dropout_2/mul_grad/Reshape6^gradients/deep_component/dropout_2/mul_grad/Reshape_1
?
Dgradients/deep_component/dropout_2/mul_grad/tuple/control_dependencyIdentity3gradients/deep_component/dropout_2/mul_grad/Reshape=^gradients/deep_component/dropout_2/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/deep_component/dropout_2/mul_grad/Reshape*'
_output_shapes
:????????? 
?
Fgradients/deep_component/dropout_2/mul_grad/tuple/control_dependency_1Identity5gradients/deep_component/dropout_2/mul_grad/Reshape_1=^gradients/deep_component/dropout_2/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/deep_component/dropout_2/mul_grad/Reshape_1*
_output_shapes
: 
s
$gradients/first_order/Sum_grad/ShapeShapefirst_order/Mul*
T0*
out_type0*
_output_shapes
:
?
#gradients/first_order/Sum_grad/SizeConst*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
"gradients/first_order/Sum_grad/addAdd!first_order/Sum/reduction_indices#gradients/first_order/Sum_grad/Size*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
_output_shapes
: 
?
"gradients/first_order/Sum_grad/modFloorMod"gradients/first_order/Sum_grad/add#gradients/first_order/Sum_grad/Size*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
_output_shapes
: 
?
&gradients/first_order/Sum_grad/Shape_1Const*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
*gradients/first_order/Sum_grad/range/startConst*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
*gradients/first_order/Sum_grad/range/deltaConst*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients/first_order/Sum_grad/rangeRange*gradients/first_order/Sum_grad/range/start#gradients/first_order/Sum_grad/Size*gradients/first_order/Sum_grad/range/delta*

Tidx0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
_output_shapes
:
?
)gradients/first_order/Sum_grad/Fill/valueConst*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
#gradients/first_order/Sum_grad/FillFill&gradients/first_order/Sum_grad/Shape_1)gradients/first_order/Sum_grad/Fill/value*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
,gradients/first_order/Sum_grad/DynamicStitchDynamicStitch$gradients/first_order/Sum_grad/range"gradients/first_order/Sum_grad/mod$gradients/first_order/Sum_grad/Shape#gradients/first_order/Sum_grad/Fill*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
N*
_output_shapes
:
?
(gradients/first_order/Sum_grad/Maximum/yConst*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
&gradients/first_order/Sum_grad/MaximumMaximum,gradients/first_order/Sum_grad/DynamicStitch(gradients/first_order/Sum_grad/Maximum/y*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
_output_shapes
:
?
'gradients/first_order/Sum_grad/floordivFloorDiv$gradients/first_order/Sum_grad/Shape&gradients/first_order/Sum_grad/Maximum*
T0*7
_class-
+)loc:@gradients/first_order/Sum_grad/Shape*
_output_shapes
:
?
&gradients/first_order/Sum_grad/ReshapeReshape?gradients/first_order/dropout/mul_grad/tuple/control_dependency,gradients/first_order/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
#gradients/first_order/Sum_grad/TileTile&gradients/first_order/Sum_grad/Reshape'gradients/first_order/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
[
gradients/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
]
gradients/mul_grad/Shape_1ShapeSub*
T0*
out_type0*
_output_shapes
:
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/mul_grad/MulMul3gradients/dropout/mul_grad/tuple/control_dependencySub*
T0*'
_output_shapes
:?????????
?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
gradients/mul_grad/Mul_1Mulmul/x3gradients/dropout/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*
_output_shapes
: 
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*'
_output_shapes
:?????????
?
-gradients/deep_component/Relu_1_grad/ReluGradReluGradDgradients/deep_component/dropout_2/mul_grad/tuple/control_dependencydeep_component/Relu_1*
T0*'
_output_shapes
:????????? 
?
$gradients/first_order/Mul_grad/ShapeShape%first_order/embedding_lookup/Identity*
T0*
out_type0*
_output_shapes
:
~
&gradients/first_order/Mul_grad/Shape_1Shapeembedding_lookup/Reshape*
T0*
out_type0*
_output_shapes
:
?
4gradients/first_order/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/first_order/Mul_grad/Shape&gradients/first_order/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
"gradients/first_order/Mul_grad/MulMul#gradients/first_order/Sum_grad/Tileembedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
?
"gradients/first_order/Mul_grad/SumSum"gradients/first_order/Mul_grad/Mul4gradients/first_order/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
&gradients/first_order/Mul_grad/ReshapeReshape"gradients/first_order/Mul_grad/Sum$gradients/first_order/Mul_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
$gradients/first_order/Mul_grad/Mul_1Mul%first_order/embedding_lookup/Identity#gradients/first_order/Sum_grad/Tile*
T0*+
_output_shapes
:?????????
?
$gradients/first_order/Mul_grad/Sum_1Sum$gradients/first_order/Mul_grad/Mul_16gradients/first_order/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients/first_order/Mul_grad/Reshape_1Reshape$gradients/first_order/Mul_grad/Sum_1&gradients/first_order/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
/gradients/first_order/Mul_grad/tuple/group_depsNoOp'^gradients/first_order/Mul_grad/Reshape)^gradients/first_order/Mul_grad/Reshape_1
?
7gradients/first_order/Mul_grad/tuple/control_dependencyIdentity&gradients/first_order/Mul_grad/Reshape0^gradients/first_order/Mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/first_order/Mul_grad/Reshape*4
_output_shapes"
 :??????????????????
?
9gradients/first_order/Mul_grad/tuple/control_dependency_1Identity(gradients/first_order/Mul_grad/Reshape_10^gradients/first_order/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/first_order/Mul_grad/Reshape_1*+
_output_shapes
:?????????
k
gradients/Sub_grad/ShapeShapesecond_order/Square*
T0*
out_type0*
_output_shapes
:
]
gradients/Sub_grad/Shape_1ShapeSum*
T0*
out_type0*
_output_shapes
:
?
(gradients/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Sub_grad/Shapegradients/Sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/Sub_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1(gradients/Sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/Sub_grad/ReshapeReshapegradients/Sub_grad/Sumgradients/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients/Sub_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1*gradients/Sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Z
gradients/Sub_grad/NegNeggradients/Sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients/Sub_grad/Reshape_1Reshapegradients/Sub_grad/Neggradients/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
g
#gradients/Sub_grad/tuple/group_depsNoOp^gradients/Sub_grad/Reshape^gradients/Sub_grad/Reshape_1
?
+gradients/Sub_grad/tuple/control_dependencyIdentitygradients/Sub_grad/Reshape$^gradients/Sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Sub_grad/Reshape*'
_output_shapes
:?????????
?
-gradients/Sub_grad/tuple/control_dependency_1Identitygradients/Sub_grad/Reshape_1$^gradients/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Sub_grad/Reshape_1*'
_output_shapes
:?????????
?
)gradients/deep_component/Add_1_grad/ShapeShapedeep_component/MatMul_1*
T0*
out_type0*
_output_shapes
:
|
+gradients/deep_component/Add_1_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
9gradients/deep_component/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/deep_component/Add_1_grad/Shape+gradients/deep_component/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
'gradients/deep_component/Add_1_grad/SumSum-gradients/deep_component/Relu_1_grad/ReluGrad9gradients/deep_component/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
+gradients/deep_component/Add_1_grad/ReshapeReshape'gradients/deep_component/Add_1_grad/Sum)gradients/deep_component/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
)gradients/deep_component/Add_1_grad/Sum_1Sum-gradients/deep_component/Relu_1_grad/ReluGrad;gradients/deep_component/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
-gradients/deep_component/Add_1_grad/Reshape_1Reshape)gradients/deep_component/Add_1_grad/Sum_1+gradients/deep_component/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
?
4gradients/deep_component/Add_1_grad/tuple/group_depsNoOp,^gradients/deep_component/Add_1_grad/Reshape.^gradients/deep_component/Add_1_grad/Reshape_1
?
<gradients/deep_component/Add_1_grad/tuple/control_dependencyIdentity+gradients/deep_component/Add_1_grad/Reshape5^gradients/deep_component/Add_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/deep_component/Add_1_grad/Reshape*'
_output_shapes
:????????? 
?
>gradients/deep_component/Add_1_grad/tuple/control_dependency_1Identity-gradients/deep_component/Add_1_grad/Reshape_15^gradients/deep_component/Add_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/deep_component/Add_1_grad/Reshape_1*
_output_shapes

: 
?
(gradients/second_order/Square_grad/ConstConst,^gradients/Sub_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
&gradients/second_order/Square_grad/MulMulsecond_order/Sum(gradients/second_order/Square_grad/Const*
T0*'
_output_shapes
:?????????
?
(gradients/second_order/Square_grad/Mul_1Mul+gradients/Sub_grad/tuple/control_dependency&gradients/second_order/Square_grad/Mul*
T0*'
_output_shapes
:?????????
^
gradients/Sum_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
?
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
?
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:
?
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
?
gradients/Sum_grad/ReshapeReshape-gradients/Sub_grad/tuple/control_dependency_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
?
-gradients/deep_component/MatMul_1_grad/MatMulMatMul<gradients/deep_component/Add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:????????? 
?
/gradients/deep_component/MatMul_1_grad/MatMul_1MatMuldeep_component/dropout_1/mul_1<gradients/deep_component/Add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:  
?
7gradients/deep_component/MatMul_1_grad/tuple/group_depsNoOp.^gradients/deep_component/MatMul_1_grad/MatMul0^gradients/deep_component/MatMul_1_grad/MatMul_1
?
?gradients/deep_component/MatMul_1_grad/tuple/control_dependencyIdentity-gradients/deep_component/MatMul_1_grad/MatMul8^gradients/deep_component/MatMul_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/deep_component/MatMul_1_grad/MatMul*'
_output_shapes
:????????? 
?
Agradients/deep_component/MatMul_1_grad/tuple/control_dependency_1Identity/gradients/deep_component/MatMul_1_grad/MatMul_18^gradients/deep_component/MatMul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/deep_component/MatMul_1_grad/MatMul_1*
_output_shapes

:  
?
1gradients/first_order/embedding_lookup_grad/ShapeConst*
_class
loc:@feature_bias*%
valueB	"?              *
dtype0	*
_output_shapes
:
?
0gradients/first_order/embedding_lookup_grad/CastCast1gradients/first_order/embedding_lookup_grad/Shape*

SrcT0	*
_class
loc:@feature_bias*
Truncate( *

DstT0*
_output_shapes
:
u
0gradients/first_order/embedding_lookup_grad/SizeSize
feat_index*
T0*
out_type0*
_output_shapes
: 
|
:gradients/first_order/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
6gradients/first_order/embedding_lookup_grad/ExpandDims
ExpandDims0gradients/first_order/embedding_lookup_grad/Size:gradients/first_order/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
?gradients/first_order/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Agradients/first_order/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Agradients/first_order/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
9gradients/first_order/embedding_lookup_grad/strided_sliceStridedSlice0gradients/first_order/embedding_lookup_grad/Cast?gradients/first_order/embedding_lookup_grad/strided_slice/stackAgradients/first_order/embedding_lookup_grad/strided_slice/stack_1Agradients/first_order/embedding_lookup_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
y
7gradients/first_order/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
2gradients/first_order/embedding_lookup_grad/concatConcatV26gradients/first_order/embedding_lookup_grad/ExpandDims9gradients/first_order/embedding_lookup_grad/strided_slice7gradients/first_order/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
3gradients/first_order/embedding_lookup_grad/ReshapeReshape7gradients/first_order/Mul_grad/tuple/control_dependency2gradients/first_order/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients/first_order/embedding_lookup_grad/Reshape_1Reshape
feat_index6gradients/first_order/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
y
%gradients/second_order/Sum_grad/ShapeShapeembedding_lookup/Mul*
T0*
out_type0*
_output_shapes
:
?
$gradients/second_order/Sum_grad/SizeConst*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
#gradients/second_order/Sum_grad/addAdd"second_order/Sum/reduction_indices$gradients/second_order/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
_output_shapes
: 
?
#gradients/second_order/Sum_grad/modFloorMod#gradients/second_order/Sum_grad/add$gradients/second_order/Sum_grad/Size*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
_output_shapes
: 
?
'gradients/second_order/Sum_grad/Shape_1Const*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
+gradients/second_order/Sum_grad/range/startConst*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
+gradients/second_order/Sum_grad/range/deltaConst*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients/second_order/Sum_grad/rangeRange+gradients/second_order/Sum_grad/range/start$gradients/second_order/Sum_grad/Size+gradients/second_order/Sum_grad/range/delta*

Tidx0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
_output_shapes
:
?
*gradients/second_order/Sum_grad/Fill/valueConst*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients/second_order/Sum_grad/FillFill'gradients/second_order/Sum_grad/Shape_1*gradients/second_order/Sum_grad/Fill/value*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
-gradients/second_order/Sum_grad/DynamicStitchDynamicStitch%gradients/second_order/Sum_grad/range#gradients/second_order/Sum_grad/mod%gradients/second_order/Sum_grad/Shape$gradients/second_order/Sum_grad/Fill*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
N*
_output_shapes
:
?
)gradients/second_order/Sum_grad/Maximum/yConst*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
'gradients/second_order/Sum_grad/MaximumMaximum-gradients/second_order/Sum_grad/DynamicStitch)gradients/second_order/Sum_grad/Maximum/y*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
_output_shapes
:
?
(gradients/second_order/Sum_grad/floordivFloorDiv%gradients/second_order/Sum_grad/Shape'gradients/second_order/Sum_grad/Maximum*
T0*8
_class.
,*loc:@gradients/second_order/Sum_grad/Shape*
_output_shapes
:
?
'gradients/second_order/Sum_grad/ReshapeReshape(gradients/second_order/Square_grad/Mul_1-gradients/second_order/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
$gradients/second_order/Sum_grad/TileTile'gradients/second_order/Sum_grad/Reshape(gradients/second_order/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
z
gradients/Square_grad/ConstConst^gradients/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
gradients/Square_grad/MulMulembedding_lookup/Mulgradients/Square_grad/Const*
T0*+
_output_shapes
:?????????
?
gradients/Square_grad/Mul_1Mulgradients/Sum_grad/Tilegradients/Square_grad/Mul*
T0*+
_output_shapes
:?????????
?
3gradients/deep_component/dropout_1/mul_1_grad/ShapeShapedeep_component/dropout_1/mul*
T0*
out_type0*
_output_shapes
:
?
5gradients/deep_component/dropout_1/mul_1_grad/Shape_1Shapedeep_component/dropout_1/Cast*
T0*
out_type0*
_output_shapes
:
?
Cgradients/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/deep_component/dropout_1/mul_1_grad/Shape5gradients/deep_component/dropout_1/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients/deep_component/dropout_1/mul_1_grad/MulMul?gradients/deep_component/MatMul_1_grad/tuple/control_dependencydeep_component/dropout_1/Cast*
T0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_1/mul_1_grad/SumSum1gradients/deep_component/dropout_1/mul_1_grad/MulCgradients/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients/deep_component/dropout_1/mul_1_grad/ReshapeReshape1gradients/deep_component/dropout_1/mul_1_grad/Sum3gradients/deep_component/dropout_1/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
3gradients/deep_component/dropout_1/mul_1_grad/Mul_1Muldeep_component/dropout_1/mul?gradients/deep_component/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
3gradients/deep_component/dropout_1/mul_1_grad/Sum_1Sum3gradients/deep_component/dropout_1/mul_1_grad/Mul_1Egradients/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients/deep_component/dropout_1/mul_1_grad/Reshape_1Reshape3gradients/deep_component/dropout_1/mul_1_grad/Sum_15gradients/deep_component/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:????????? 
?
>gradients/deep_component/dropout_1/mul_1_grad/tuple/group_depsNoOp6^gradients/deep_component/dropout_1/mul_1_grad/Reshape8^gradients/deep_component/dropout_1/mul_1_grad/Reshape_1
?
Fgradients/deep_component/dropout_1/mul_1_grad/tuple/control_dependencyIdentity5gradients/deep_component/dropout_1/mul_1_grad/Reshape?^gradients/deep_component/dropout_1/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/deep_component/dropout_1/mul_1_grad/Reshape*'
_output_shapes
:????????? 
?
Hgradients/deep_component/dropout_1/mul_1_grad/tuple/control_dependency_1Identity7gradients/deep_component/dropout_1/mul_1_grad/Reshape_1?^gradients/deep_component/dropout_1/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/deep_component/dropout_1/mul_1_grad/Reshape_1*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_1/mul_grad/ShapeShapedeep_component/Relu*
T0*
out_type0*
_output_shapes
:
v
3gradients/deep_component/dropout_1/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Agradients/deep_component/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/deep_component/dropout_1/mul_grad/Shape3gradients/deep_component/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients/deep_component/dropout_1/mul_grad/MulMulFgradients/deep_component/dropout_1/mul_1_grad/tuple/control_dependency deep_component/dropout_1/truediv*
T0*'
_output_shapes
:????????? 
?
/gradients/deep_component/dropout_1/mul_grad/SumSum/gradients/deep_component/dropout_1/mul_grad/MulAgradients/deep_component/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
3gradients/deep_component/dropout_1/mul_grad/ReshapeReshape/gradients/deep_component/dropout_1/mul_grad/Sum1gradients/deep_component/dropout_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_1/mul_grad/Mul_1Muldeep_component/ReluFgradients/deep_component/dropout_1/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
1gradients/deep_component/dropout_1/mul_grad/Sum_1Sum1gradients/deep_component/dropout_1/mul_grad/Mul_1Cgradients/deep_component/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients/deep_component/dropout_1/mul_grad/Reshape_1Reshape1gradients/deep_component/dropout_1/mul_grad/Sum_13gradients/deep_component/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
<gradients/deep_component/dropout_1/mul_grad/tuple/group_depsNoOp4^gradients/deep_component/dropout_1/mul_grad/Reshape6^gradients/deep_component/dropout_1/mul_grad/Reshape_1
?
Dgradients/deep_component/dropout_1/mul_grad/tuple/control_dependencyIdentity3gradients/deep_component/dropout_1/mul_grad/Reshape=^gradients/deep_component/dropout_1/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/deep_component/dropout_1/mul_grad/Reshape*'
_output_shapes
:????????? 
?
Fgradients/deep_component/dropout_1/mul_grad/tuple/control_dependency_1Identity5gradients/deep_component/dropout_1/mul_grad/Reshape_1=^gradients/deep_component/dropout_1/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/deep_component/dropout_1/mul_grad/Reshape_1*
_output_shapes
: 
?
+gradients/deep_component/Relu_grad/ReluGradReluGradDgradients/deep_component/dropout_1/mul_grad/tuple/control_dependencydeep_component/Relu*
T0*'
_output_shapes
:????????? 
|
'gradients/deep_component/Add_grad/ShapeShapedeep_component/MatMul*
T0*
out_type0*
_output_shapes
:
z
)gradients/deep_component/Add_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
7gradients/deep_component/Add_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/deep_component/Add_grad/Shape)gradients/deep_component/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
%gradients/deep_component/Add_grad/SumSum+gradients/deep_component/Relu_grad/ReluGrad7gradients/deep_component/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
)gradients/deep_component/Add_grad/ReshapeReshape%gradients/deep_component/Add_grad/Sum'gradients/deep_component/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
'gradients/deep_component/Add_grad/Sum_1Sum+gradients/deep_component/Relu_grad/ReluGrad9gradients/deep_component/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
+gradients/deep_component/Add_grad/Reshape_1Reshape'gradients/deep_component/Add_grad/Sum_1)gradients/deep_component/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
?
2gradients/deep_component/Add_grad/tuple/group_depsNoOp*^gradients/deep_component/Add_grad/Reshape,^gradients/deep_component/Add_grad/Reshape_1
?
:gradients/deep_component/Add_grad/tuple/control_dependencyIdentity)gradients/deep_component/Add_grad/Reshape3^gradients/deep_component/Add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/deep_component/Add_grad/Reshape*'
_output_shapes
:????????? 
?
<gradients/deep_component/Add_grad/tuple/control_dependency_1Identity+gradients/deep_component/Add_grad/Reshape_13^gradients/deep_component/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/deep_component/Add_grad/Reshape_1*
_output_shapes

: 
?
+gradients/deep_component/MatMul_grad/MatMulMatMul:gradients/deep_component/Add_grad/tuple/control_dependencyVariable/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
-gradients/deep_component/MatMul_grad/MatMul_1MatMuldeep_component/dropout/mul_1:gradients/deep_component/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	? 
?
5gradients/deep_component/MatMul_grad/tuple/group_depsNoOp,^gradients/deep_component/MatMul_grad/MatMul.^gradients/deep_component/MatMul_grad/MatMul_1
?
=gradients/deep_component/MatMul_grad/tuple/control_dependencyIdentity+gradients/deep_component/MatMul_grad/MatMul6^gradients/deep_component/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/deep_component/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
?gradients/deep_component/MatMul_grad/tuple/control_dependency_1Identity-gradients/deep_component/MatMul_grad/MatMul_16^gradients/deep_component/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/deep_component/MatMul_grad/MatMul_1*
_output_shapes
:	? 
?
1gradients/deep_component/dropout/mul_1_grad/ShapeShapedeep_component/dropout/mul*
T0*
out_type0*
_output_shapes
:
?
3gradients/deep_component/dropout/mul_1_grad/Shape_1Shapedeep_component/dropout/Cast*
T0*
out_type0*
_output_shapes
:
?
Agradients/deep_component/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/deep_component/dropout/mul_1_grad/Shape3gradients/deep_component/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients/deep_component/dropout/mul_1_grad/MulMul=gradients/deep_component/MatMul_grad/tuple/control_dependencydeep_component/dropout/Cast*
T0*(
_output_shapes
:??????????
?
/gradients/deep_component/dropout/mul_1_grad/SumSum/gradients/deep_component/dropout/mul_1_grad/MulAgradients/deep_component/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
3gradients/deep_component/dropout/mul_1_grad/ReshapeReshape/gradients/deep_component/dropout/mul_1_grad/Sum1gradients/deep_component/dropout/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
1gradients/deep_component/dropout/mul_1_grad/Mul_1Muldeep_component/dropout/mul=gradients/deep_component/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
1gradients/deep_component/dropout/mul_1_grad/Sum_1Sum1gradients/deep_component/dropout/mul_1_grad/Mul_1Cgradients/deep_component/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients/deep_component/dropout/mul_1_grad/Reshape_1Reshape1gradients/deep_component/dropout/mul_1_grad/Sum_13gradients/deep_component/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:??????????
?
<gradients/deep_component/dropout/mul_1_grad/tuple/group_depsNoOp4^gradients/deep_component/dropout/mul_1_grad/Reshape6^gradients/deep_component/dropout/mul_1_grad/Reshape_1
?
Dgradients/deep_component/dropout/mul_1_grad/tuple/control_dependencyIdentity3gradients/deep_component/dropout/mul_1_grad/Reshape=^gradients/deep_component/dropout/mul_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/deep_component/dropout/mul_1_grad/Reshape*(
_output_shapes
:??????????
?
Fgradients/deep_component/dropout/mul_1_grad/tuple/control_dependency_1Identity5gradients/deep_component/dropout/mul_1_grad/Reshape_1=^gradients/deep_component/dropout/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/deep_component/dropout/mul_1_grad/Reshape_1*(
_output_shapes
:??????????
?
/gradients/deep_component/dropout/mul_grad/ShapeShapedeep_component/Reshape*
T0*
out_type0*
_output_shapes
:
t
1gradients/deep_component/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
?gradients/deep_component/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/deep_component/dropout/mul_grad/Shape1gradients/deep_component/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
-gradients/deep_component/dropout/mul_grad/MulMulDgradients/deep_component/dropout/mul_1_grad/tuple/control_dependencydeep_component/dropout/truediv*
T0*(
_output_shapes
:??????????
?
-gradients/deep_component/dropout/mul_grad/SumSum-gradients/deep_component/dropout/mul_grad/Mul?gradients/deep_component/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
1gradients/deep_component/dropout/mul_grad/ReshapeReshape-gradients/deep_component/dropout/mul_grad/Sum/gradients/deep_component/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
/gradients/deep_component/dropout/mul_grad/Mul_1Muldeep_component/ReshapeDgradients/deep_component/dropout/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
/gradients/deep_component/dropout/mul_grad/Sum_1Sum/gradients/deep_component/dropout/mul_grad/Mul_1Agradients/deep_component/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
3gradients/deep_component/dropout/mul_grad/Reshape_1Reshape/gradients/deep_component/dropout/mul_grad/Sum_11gradients/deep_component/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
:gradients/deep_component/dropout/mul_grad/tuple/group_depsNoOp2^gradients/deep_component/dropout/mul_grad/Reshape4^gradients/deep_component/dropout/mul_grad/Reshape_1
?
Bgradients/deep_component/dropout/mul_grad/tuple/control_dependencyIdentity1gradients/deep_component/dropout/mul_grad/Reshape;^gradients/deep_component/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/deep_component/dropout/mul_grad/Reshape*(
_output_shapes
:??????????
?
Dgradients/deep_component/dropout/mul_grad/tuple/control_dependency_1Identity3gradients/deep_component/dropout/mul_grad/Reshape_1;^gradients/deep_component/dropout/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/deep_component/dropout/mul_grad/Reshape_1*
_output_shapes
: 

+gradients/deep_component/Reshape_grad/ShapeShapeembedding_lookup/Mul*
T0*
out_type0*
_output_shapes
:
?
-gradients/deep_component/Reshape_grad/ReshapeReshapeBgradients/deep_component/dropout/mul_grad/tuple/control_dependency+gradients/deep_component/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
gradients/AddN_1AddN$gradients/second_order/Sum_grad/Tilegradients/Square_grad/Mul_1-gradients/deep_component/Reshape_grad/Reshape*
T0*7
_class-
+)loc:@gradients/second_order/Sum_grad/Tile*
N*+
_output_shapes
:?????????
?
)gradients/embedding_lookup/Mul_grad/ShapeShape*embedding_lookup/embedding_lookup/Identity*
T0*
out_type0*
_output_shapes
:
?
+gradients/embedding_lookup/Mul_grad/Shape_1Shapeembedding_lookup/Reshape*
T0*
out_type0*
_output_shapes
:
?
9gradients/embedding_lookup/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients/embedding_lookup/Mul_grad/Shape+gradients/embedding_lookup/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
'gradients/embedding_lookup/Mul_grad/MulMulgradients/AddN_1embedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
?
'gradients/embedding_lookup/Mul_grad/SumSum'gradients/embedding_lookup/Mul_grad/Mul9gradients/embedding_lookup/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
+gradients/embedding_lookup/Mul_grad/ReshapeReshape'gradients/embedding_lookup/Mul_grad/Sum)gradients/embedding_lookup/Mul_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
)gradients/embedding_lookup/Mul_grad/Mul_1Mul*embedding_lookup/embedding_lookup/Identitygradients/AddN_1*
T0*+
_output_shapes
:?????????
?
)gradients/embedding_lookup/Mul_grad/Sum_1Sum)gradients/embedding_lookup/Mul_grad/Mul_1;gradients/embedding_lookup/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
-gradients/embedding_lookup/Mul_grad/Reshape_1Reshape)gradients/embedding_lookup/Mul_grad/Sum_1+gradients/embedding_lookup/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
4gradients/embedding_lookup/Mul_grad/tuple/group_depsNoOp,^gradients/embedding_lookup/Mul_grad/Reshape.^gradients/embedding_lookup/Mul_grad/Reshape_1
?
<gradients/embedding_lookup/Mul_grad/tuple/control_dependencyIdentity+gradients/embedding_lookup/Mul_grad/Reshape5^gradients/embedding_lookup/Mul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/embedding_lookup/Mul_grad/Reshape*4
_output_shapes"
 :??????????????????
?
>gradients/embedding_lookup/Mul_grad/tuple/control_dependency_1Identity-gradients/embedding_lookup/Mul_grad/Reshape_15^gradients/embedding_lookup/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/embedding_lookup/Mul_grad/Reshape_1*+
_output_shapes
:?????????
?
6gradients/embedding_lookup/embedding_lookup_grad/ShapeConst*%
_class
loc:@feature_embeddings*%
valueB	"?              *
dtype0	*
_output_shapes
:
?
5gradients/embedding_lookup/embedding_lookup_grad/CastCast6gradients/embedding_lookup/embedding_lookup_grad/Shape*

SrcT0	*%
_class
loc:@feature_embeddings*
Truncate( *

DstT0*
_output_shapes
:
z
5gradients/embedding_lookup/embedding_lookup_grad/SizeSize
feat_index*
T0*
out_type0*
_output_shapes
: 
?
?gradients/embedding_lookup/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
;gradients/embedding_lookup/embedding_lookup_grad/ExpandDims
ExpandDims5gradients/embedding_lookup/embedding_lookup_grad/Size?gradients/embedding_lookup/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Dgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Fgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Fgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
>gradients/embedding_lookup/embedding_lookup_grad/strided_sliceStridedSlice5gradients/embedding_lookup/embedding_lookup_grad/CastDgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stackFgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stack_1Fgradients/embedding_lookup/embedding_lookup_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
~
<gradients/embedding_lookup/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
7gradients/embedding_lookup/embedding_lookup_grad/concatConcatV2;gradients/embedding_lookup/embedding_lookup_grad/ExpandDims>gradients/embedding_lookup/embedding_lookup_grad/strided_slice<gradients/embedding_lookup/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
8gradients/embedding_lookup/embedding_lookup_grad/ReshapeReshape<gradients/embedding_lookup/Mul_grad/tuple/control_dependency7gradients/embedding_lookup/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
:gradients/embedding_lookup/embedding_lookup_grad/Reshape_1Reshape
feat_index;gradients/embedding_lookup/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
{
beta1_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_class
loc:@Variable*
valueB
 *w??*
dtype0*
_output_shapes
: 
?
beta2_power
VariableV2*
shape: *
shared_name *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
: 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
9feature_embeddings/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"?      *%
_class
loc:@feature_embeddings*
dtype0*
_output_shapes
:
?
/feature_embeddings/Adam/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@feature_embeddings*
dtype0*
_output_shapes
: 
?
)feature_embeddings/Adam/Initializer/zerosFill9feature_embeddings/Adam/Initializer/zeros/shape_as_tensor/feature_embeddings/Adam/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
feature_embeddings/Adam
VariableV2*
shape:	?*
shared_name *%
_class
loc:@feature_embeddings*
dtype0*
	container *
_output_shapes
:	?
?
feature_embeddings/Adam/AssignAssignfeature_embeddings/Adam)feature_embeddings/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
feature_embeddings/Adam/readIdentityfeature_embeddings/Adam*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
;feature_embeddings/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"?      *%
_class
loc:@feature_embeddings*
dtype0*
_output_shapes
:
?
1feature_embeddings/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *%
_class
loc:@feature_embeddings*
dtype0*
_output_shapes
: 
?
+feature_embeddings/Adam_1/Initializer/zerosFill;feature_embeddings/Adam_1/Initializer/zeros/shape_as_tensor1feature_embeddings/Adam_1/Initializer/zeros/Const*
T0*

index_type0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
feature_embeddings/Adam_1
VariableV2*
shape:	?*
shared_name *%
_class
loc:@feature_embeddings*
dtype0*
	container *
_output_shapes
:	?
?
 feature_embeddings/Adam_1/AssignAssignfeature_embeddings/Adam_1+feature_embeddings/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
feature_embeddings/Adam_1/readIdentityfeature_embeddings/Adam_1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
#feature_bias/Adam/Initializer/zerosConst*
valueB	?*    *
_class
loc:@feature_bias*
dtype0*
_output_shapes
:	?
?
feature_bias/Adam
VariableV2*
shape:	?*
shared_name *
_class
loc:@feature_bias*
dtype0*
	container *
_output_shapes
:	?
?
feature_bias/Adam/AssignAssignfeature_bias/Adam#feature_bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
feature_bias/Adam/readIdentityfeature_bias/Adam*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
%feature_bias/Adam_1/Initializer/zerosConst*
valueB	?*    *
_class
loc:@feature_bias*
dtype0*
_output_shapes
:	?
?
feature_bias/Adam_1
VariableV2*
shape:	?*
shared_name *
_class
loc:@feature_bias*
dtype0*
	container *
_output_shapes
:	?
?
feature_bias/Adam_1/AssignAssignfeature_bias/Adam_1%feature_bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
feature_bias/Adam_1/readIdentityfeature_bias/Adam_1*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"?       *
_class
loc:@Variable*
dtype0*
_output_shapes
:
?
%Variable/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
?
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*
_output_shapes
:	? 
?
Variable/Adam
VariableV2*
shape:	? *
shared_name *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
:	? 
?
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
t
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes
:	? 
?
1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"?       *
_class
loc:@Variable*
dtype0*
_output_shapes
:
?
'Variable/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
?
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*
_output_shapes
:	? 
?
Variable/Adam_1
VariableV2*
shape:	? *
shared_name *
_class
loc:@Variable*
dtype0*
	container *
_output_shapes
:	? 
?
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
x
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes
:	? 
?
!Variable_1/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes

: 
?
Variable_1/Adam
VariableV2*
shape
: *
shared_name *
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes

: 
?
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes

: 
?
#Variable_1/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes

: 
?
Variable_1/Adam_1
VariableV2*
shape
: *
shared_name *
_class
loc:@Variable_1*
dtype0*
	container *
_output_shapes

: 
?
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
}
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes

: 
?
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"        *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:
?
'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
?
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*
_output_shapes

:  
?
Variable_2/Adam
VariableV2*
shape
:  *
shared_name *
_class
loc:@Variable_2*
dtype0*
	container *
_output_shapes

:  
?
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:  
?
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"        *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:
?
)Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
?
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*
_output_shapes

:  
?
Variable_2/Adam_1
VariableV2*
shape
:  *
shared_name *
_class
loc:@Variable_2*
dtype0*
	container *
_output_shapes

:  
?
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:  
?
!Variable_3/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes

: 
?
Variable_3/Adam
VariableV2*
shape
: *
shared_name *
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes

: 
?
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes

: 
?
#Variable_3/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes

: 
?
Variable_3/Adam_1
VariableV2*
shape
: *
shared_name *
_class
loc:@Variable_3*
dtype0*
	container *
_output_shapes

: 
?
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
}
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes

: 
?
!Variable_4/Adam/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_4*
dtype0*
_output_shapes

:<
?
Variable_4/Adam
VariableV2*
shape
:<*
shared_name *
_class
loc:@Variable_4*
dtype0*
	container *
_output_shapes

:<
?
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:<
?
#Variable_4/Adam_1/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_4*
dtype0*
_output_shapes

:<
?
Variable_4/Adam_1
VariableV2*
shape
:<*
shared_name *
_class
loc:@Variable_4*
dtype0*
	container *
_output_shapes

:<
?
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*
_output_shapes

:<
?
!Variable_5/Adam/Initializer/zerosConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
?
Variable_5/Adam
VariableV2*
shape: *
shared_name *
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes
: 
?
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
q
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
: 
?
#Variable_5/Adam_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
?
Variable_5/Adam_1
VariableV2*
shape: *
shared_name *
_class
loc:@Variable_5*
dtype0*
	container *
_output_shapes
: 
?
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
u
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
: 
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
%Adam/update_feature_embeddings/UniqueUnique:gradients/embedding_lookup/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*%
_class
loc:@feature_embeddings*2
_output_shapes 
:?????????:?????????
?
$Adam/update_feature_embeddings/ShapeShape%Adam/update_feature_embeddings/Unique*
T0*%
_class
loc:@feature_embeddings*
out_type0*
_output_shapes
:
?
2Adam/update_feature_embeddings/strided_slice/stackConst*%
_class
loc:@feature_embeddings*
valueB: *
dtype0*
_output_shapes
:
?
4Adam/update_feature_embeddings/strided_slice/stack_1Const*%
_class
loc:@feature_embeddings*
valueB:*
dtype0*
_output_shapes
:
?
4Adam/update_feature_embeddings/strided_slice/stack_2Const*%
_class
loc:@feature_embeddings*
valueB:*
dtype0*
_output_shapes
:
?
,Adam/update_feature_embeddings/strided_sliceStridedSlice$Adam/update_feature_embeddings/Shape2Adam/update_feature_embeddings/strided_slice/stack4Adam/update_feature_embeddings/strided_slice/stack_14Adam/update_feature_embeddings/strided_slice/stack_2*
T0*
Index0*%
_class
loc:@feature_embeddings*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
1Adam/update_feature_embeddings/UnsortedSegmentSumUnsortedSegmentSum8gradients/embedding_lookup/embedding_lookup_grad/Reshape'Adam/update_feature_embeddings/Unique:1,Adam/update_feature_embeddings/strided_slice*
Tnumsegments0*
Tindices0*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
$Adam/update_feature_embeddings/sub/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
"Adam/update_feature_embeddings/subSub$Adam/update_feature_embeddings/sub/xbeta2_power/read*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
#Adam/update_feature_embeddings/SqrtSqrt"Adam/update_feature_embeddings/sub*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
"Adam/update_feature_embeddings/mulMulAdam/learning_rate#Adam/update_feature_embeddings/Sqrt*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
&Adam/update_feature_embeddings/sub_1/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
$Adam/update_feature_embeddings/sub_1Sub&Adam/update_feature_embeddings/sub_1/xbeta1_power/read*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
&Adam/update_feature_embeddings/truedivRealDiv"Adam/update_feature_embeddings/mul$Adam/update_feature_embeddings/sub_1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
&Adam/update_feature_embeddings/sub_2/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
$Adam/update_feature_embeddings/sub_2Sub&Adam/update_feature_embeddings/sub_2/x
Adam/beta1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
$Adam/update_feature_embeddings/mul_1Mul1Adam/update_feature_embeddings/UnsortedSegmentSum$Adam/update_feature_embeddings/sub_2*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
$Adam/update_feature_embeddings/mul_2Mulfeature_embeddings/Adam/read
Adam/beta1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
%Adam/update_feature_embeddings/AssignAssignfeature_embeddings/Adam$Adam/update_feature_embeddings/mul_2*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
)Adam/update_feature_embeddings/ScatterAdd
ScatterAddfeature_embeddings/Adam%Adam/update_feature_embeddings/Unique$Adam/update_feature_embeddings/mul_1&^Adam/update_feature_embeddings/Assign*
use_locking( *
Tindices0*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
$Adam/update_feature_embeddings/mul_3Mul1Adam/update_feature_embeddings/UnsortedSegmentSum1Adam/update_feature_embeddings/UnsortedSegmentSum*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
&Adam/update_feature_embeddings/sub_3/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
$Adam/update_feature_embeddings/sub_3Sub&Adam/update_feature_embeddings/sub_3/x
Adam/beta2*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
$Adam/update_feature_embeddings/mul_4Mul$Adam/update_feature_embeddings/mul_3$Adam/update_feature_embeddings/sub_3*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
$Adam/update_feature_embeddings/mul_5Mulfeature_embeddings/Adam_1/read
Adam/beta2*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
'Adam/update_feature_embeddings/Assign_1Assignfeature_embeddings/Adam_1$Adam/update_feature_embeddings/mul_5*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
+Adam/update_feature_embeddings/ScatterAdd_1
ScatterAddfeature_embeddings/Adam_1%Adam/update_feature_embeddings/Unique$Adam/update_feature_embeddings/mul_4(^Adam/update_feature_embeddings/Assign_1*
use_locking( *
Tindices0*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
%Adam/update_feature_embeddings/Sqrt_1Sqrt+Adam/update_feature_embeddings/ScatterAdd_1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
$Adam/update_feature_embeddings/mul_6Mul&Adam/update_feature_embeddings/truediv)Adam/update_feature_embeddings/ScatterAdd*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
"Adam/update_feature_embeddings/addAdd%Adam/update_feature_embeddings/Sqrt_1Adam/epsilon*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
(Adam/update_feature_embeddings/truediv_1RealDiv$Adam/update_feature_embeddings/mul_6"Adam/update_feature_embeddings/add*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
(Adam/update_feature_embeddings/AssignSub	AssignSubfeature_embeddings(Adam/update_feature_embeddings/truediv_1*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
)Adam/update_feature_embeddings/group_depsNoOp)^Adam/update_feature_embeddings/AssignSub*^Adam/update_feature_embeddings/ScatterAdd,^Adam/update_feature_embeddings/ScatterAdd_1*%
_class
loc:@feature_embeddings
?
Adam/update_feature_bias/UniqueUnique5gradients/first_order/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*
_class
loc:@feature_bias*2
_output_shapes 
:?????????:?????????
?
Adam/update_feature_bias/ShapeShapeAdam/update_feature_bias/Unique*
T0*
_class
loc:@feature_bias*
out_type0*
_output_shapes
:
?
,Adam/update_feature_bias/strided_slice/stackConst*
_class
loc:@feature_bias*
valueB: *
dtype0*
_output_shapes
:
?
.Adam/update_feature_bias/strided_slice/stack_1Const*
_class
loc:@feature_bias*
valueB:*
dtype0*
_output_shapes
:
?
.Adam/update_feature_bias/strided_slice/stack_2Const*
_class
loc:@feature_bias*
valueB:*
dtype0*
_output_shapes
:
?
&Adam/update_feature_bias/strided_sliceStridedSliceAdam/update_feature_bias/Shape,Adam/update_feature_bias/strided_slice/stack.Adam/update_feature_bias/strided_slice/stack_1.Adam/update_feature_bias/strided_slice/stack_2*
T0*
Index0*
_class
loc:@feature_bias*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
+Adam/update_feature_bias/UnsortedSegmentSumUnsortedSegmentSum3gradients/first_order/embedding_lookup_grad/Reshape!Adam/update_feature_bias/Unique:1&Adam/update_feature_bias/strided_slice*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
Adam/update_feature_bias/sub/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Adam/update_feature_bias/subSubAdam/update_feature_bias/sub/xbeta2_power/read*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam/update_feature_bias/SqrtSqrtAdam/update_feature_bias/sub*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam/update_feature_bias/mulMulAdam/learning_rateAdam/update_feature_bias/Sqrt*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
 Adam/update_feature_bias/sub_1/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Adam/update_feature_bias/sub_1Sub Adam/update_feature_bias/sub_1/xbeta1_power/read*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
 Adam/update_feature_bias/truedivRealDivAdam/update_feature_bias/mulAdam/update_feature_bias/sub_1*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
 Adam/update_feature_bias/sub_2/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Adam/update_feature_bias/sub_2Sub Adam/update_feature_bias/sub_2/x
Adam/beta1*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam/update_feature_bias/mul_1Mul+Adam/update_feature_bias/UnsortedSegmentSumAdam/update_feature_bias/sub_2*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
Adam/update_feature_bias/mul_2Mulfeature_bias/Adam/read
Adam/beta1*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam/update_feature_bias/AssignAssignfeature_bias/AdamAdam/update_feature_bias/mul_2*
use_locking( *
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
#Adam/update_feature_bias/ScatterAdd
ScatterAddfeature_bias/AdamAdam/update_feature_bias/UniqueAdam/update_feature_bias/mul_1 ^Adam/update_feature_bias/Assign*
use_locking( *
Tindices0*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam/update_feature_bias/mul_3Mul+Adam/update_feature_bias/UnsortedSegmentSum+Adam/update_feature_bias/UnsortedSegmentSum*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
 Adam/update_feature_bias/sub_3/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Adam/update_feature_bias/sub_3Sub Adam/update_feature_bias/sub_3/x
Adam/beta2*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam/update_feature_bias/mul_4MulAdam/update_feature_bias/mul_3Adam/update_feature_bias/sub_3*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
Adam/update_feature_bias/mul_5Mulfeature_bias/Adam_1/read
Adam/beta2*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
!Adam/update_feature_bias/Assign_1Assignfeature_bias/Adam_1Adam/update_feature_bias/mul_5*
use_locking( *
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
%Adam/update_feature_bias/ScatterAdd_1
ScatterAddfeature_bias/Adam_1Adam/update_feature_bias/UniqueAdam/update_feature_bias/mul_4"^Adam/update_feature_bias/Assign_1*
use_locking( *
Tindices0*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam/update_feature_bias/Sqrt_1Sqrt%Adam/update_feature_bias/ScatterAdd_1*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam/update_feature_bias/mul_6Mul Adam/update_feature_bias/truediv#Adam/update_feature_bias/ScatterAdd*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam/update_feature_bias/addAddAdam/update_feature_bias/Sqrt_1Adam/epsilon*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
"Adam/update_feature_bias/truediv_1RealDivAdam/update_feature_bias/mul_6Adam/update_feature_bias/add*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
"Adam/update_feature_bias/AssignSub	AssignSubfeature_bias"Adam/update_feature_bias/truediv_1*
use_locking( *
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
#Adam/update_feature_bias/group_depsNoOp#^Adam/update_feature_bias/AssignSub$^Adam/update_feature_bias/ScatterAdd&^Adam/update_feature_bias/ScatterAdd_1*
_class
loc:@feature_bias
?
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/deep_component/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes
:	? 
?
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/deep_component/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes

: 
?
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/deep_component/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:  
?
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/deep_component/Add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes

: 
?
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon4gradients/output/Mul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:<
?
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon4gradients/output/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes
: 
?
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam$^Adam/update_feature_bias/group_deps*^Adam/update_feature_embeddings/group_deps*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam$^Adam/update_feature_bias/group_deps*^Adam/update_feature_embeddings/group_deps*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam$^Adam/update_feature_bias/group_deps*^Adam/update_feature_embeddings/group_deps
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
m
*gradients_1/loss/log_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
o
,gradients_1/loss/log_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients_1/loss/log_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/value_grad/Shape,gradients_1/loss/log_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients_1/loss/log_loss/value_grad/div_no_nanDivNoNangradients_1/Fillloss/log_loss/num_present*
T0*
_output_shapes
: 
?
(gradients_1/loss/log_loss/value_grad/SumSum/gradients_1/loss/log_loss/value_grad/div_no_nan:gradients_1/loss/log_loss/value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
,gradients_1/loss/log_loss/value_grad/ReshapeReshape(gradients_1/loss/log_loss/value_grad/Sum*gradients_1/loss/log_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
e
(gradients_1/loss/log_loss/value_grad/NegNegloss/log_loss/Sum_1*
T0*
_output_shapes
: 
?
1gradients_1/loss/log_loss/value_grad/div_no_nan_1DivNoNan(gradients_1/loss/log_loss/value_grad/Negloss/log_loss/num_present*
T0*
_output_shapes
: 
?
1gradients_1/loss/log_loss/value_grad/div_no_nan_2DivNoNan1gradients_1/loss/log_loss/value_grad/div_no_nan_1loss/log_loss/num_present*
T0*
_output_shapes
: 
?
(gradients_1/loss/log_loss/value_grad/mulMulgradients_1/Fill1gradients_1/loss/log_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
?
*gradients_1/loss/log_loss/value_grad/Sum_1Sum(gradients_1/loss/log_loss/value_grad/mul<gradients_1/loss/log_loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
.gradients_1/loss/log_loss/value_grad/Reshape_1Reshape*gradients_1/loss/log_loss/value_grad/Sum_1,gradients_1/loss/log_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients_1/loss/log_loss/value_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/value_grad/Reshape/^gradients_1/loss/log_loss/value_grad/Reshape_1
?
=gradients_1/loss/log_loss/value_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/value_grad/Reshape6^gradients_1/loss/log_loss/value_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/value_grad/Reshape*
_output_shapes
: 
?
?gradients_1/loss/log_loss/value_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/value_grad/Reshape_16^gradients_1/loss/log_loss/value_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/value_grad/Reshape_1*
_output_shapes
: 
u
2gradients_1/loss/log_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
,gradients_1/loss/log_loss/Sum_1_grad/ReshapeReshape=gradients_1/loss/log_loss/value_grad/tuple/control_dependency2gradients_1/loss/log_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
m
*gradients_1/loss/log_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
)gradients_1/loss/log_loss/Sum_1_grad/TileTile,gradients_1/loss/log_loss/Sum_1_grad/Reshape*gradients_1/loss/log_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
?
0gradients_1/loss/log_loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
*gradients_1/loss/log_loss/Sum_grad/ReshapeReshape)gradients_1/loss/log_loss/Sum_1_grad/Tile0gradients_1/loss/log_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
{
(gradients_1/loss/log_loss/Sum_grad/ShapeShapeloss/log_loss/Mul_2*
T0*
out_type0*
_output_shapes
:
?
'gradients_1/loss/log_loss/Sum_grad/TileTile*gradients_1/loss/log_loss/Sum_grad/Reshape(gradients_1/loss/log_loss/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:?????????
}
*gradients_1/loss/log_loss/Mul_2_grad/ShapeShapeloss/log_loss/sub_2*
T0*
out_type0*
_output_shapes
:
o
,gradients_1/loss/log_loss/Mul_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients_1/loss/log_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/Mul_2_grad/Shape,gradients_1/loss/log_loss/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients_1/loss/log_loss/Mul_2_grad/MulMul'gradients_1/loss/log_loss/Sum_grad/Tileloss/log_loss/Cast/x*
T0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/Mul_2_grad/SumSum(gradients_1/loss/log_loss/Mul_2_grad/Mul:gradients_1/loss/log_loss/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/Mul_2_grad/ReshapeReshape(gradients_1/loss/log_loss/Mul_2_grad/Sum*gradients_1/loss/log_loss/Mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/Mul_2_grad/Mul_1Mulloss/log_loss/sub_2'gradients_1/loss/log_loss/Sum_grad/Tile*
T0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/Mul_2_grad/Sum_1Sum*gradients_1/loss/log_loss/Mul_2_grad/Mul_1<gradients_1/loss/log_loss/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients_1/loss/log_loss/Mul_2_grad/Reshape_1Reshape*gradients_1/loss/log_loss/Mul_2_grad/Sum_1,gradients_1/loss/log_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients_1/loss/log_loss/Mul_2_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/Mul_2_grad/Reshape/^gradients_1/loss/log_loss/Mul_2_grad/Reshape_1
?
=gradients_1/loss/log_loss/Mul_2_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/Mul_2_grad/Reshape6^gradients_1/loss/log_loss/Mul_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/Mul_2_grad/Reshape*'
_output_shapes
:?????????
?
?gradients_1/loss/log_loss/Mul_2_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/Mul_2_grad/Reshape_16^gradients_1/loss/log_loss/Mul_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/Mul_2_grad/Reshape_1*
_output_shapes
: 
{
*gradients_1/loss/log_loss/sub_2_grad/ShapeShapeloss/log_loss/Neg*
T0*
out_type0*
_output_shapes
:

,gradients_1/loss/log_loss/sub_2_grad/Shape_1Shapeloss/log_loss/Mul_1*
T0*
out_type0*
_output_shapes
:
?
:gradients_1/loss/log_loss/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/sub_2_grad/Shape,gradients_1/loss/log_loss/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients_1/loss/log_loss/sub_2_grad/SumSum=gradients_1/loss/log_loss/Mul_2_grad/tuple/control_dependency:gradients_1/loss/log_loss/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/sub_2_grad/ReshapeReshape(gradients_1/loss/log_loss/sub_2_grad/Sum*gradients_1/loss/log_loss/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/sub_2_grad/Sum_1Sum=gradients_1/loss/log_loss/Mul_2_grad/tuple/control_dependency<gradients_1/loss/log_loss/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
~
(gradients_1/loss/log_loss/sub_2_grad/NegNeg*gradients_1/loss/log_loss/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
.gradients_1/loss/log_loss/sub_2_grad/Reshape_1Reshape(gradients_1/loss/log_loss/sub_2_grad/Neg,gradients_1/loss/log_loss/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients_1/loss/log_loss/sub_2_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/sub_2_grad/Reshape/^gradients_1/loss/log_loss/sub_2_grad/Reshape_1
?
=gradients_1/loss/log_loss/sub_2_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/sub_2_grad/Reshape6^gradients_1/loss/log_loss/sub_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/sub_2_grad/Reshape*'
_output_shapes
:?????????
?
?gradients_1/loss/log_loss/sub_2_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/sub_2_grad/Reshape_16^gradients_1/loss/log_loss/sub_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/sub_2_grad/Reshape_1*'
_output_shapes
:?????????
?
&gradients_1/loss/log_loss/Neg_grad/NegNeg=gradients_1/loss/log_loss/sub_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
{
*gradients_1/loss/log_loss/Mul_1_grad/ShapeShapeloss/log_loss/sub*
T0*
out_type0*
_output_shapes
:

,gradients_1/loss/log_loss/Mul_1_grad/Shape_1Shapeloss/log_loss/Log_1*
T0*
out_type0*
_output_shapes
:
?
:gradients_1/loss/log_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/Mul_1_grad/Shape,gradients_1/loss/log_loss/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients_1/loss/log_loss/Mul_1_grad/MulMul?gradients_1/loss/log_loss/sub_2_grad/tuple/control_dependency_1loss/log_loss/Log_1*
T0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/Mul_1_grad/SumSum(gradients_1/loss/log_loss/Mul_1_grad/Mul:gradients_1/loss/log_loss/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/Mul_1_grad/ReshapeReshape(gradients_1/loss/log_loss/Mul_1_grad/Sum*gradients_1/loss/log_loss/Mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/Mul_1_grad/Mul_1Mulloss/log_loss/sub?gradients_1/loss/log_loss/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/Mul_1_grad/Sum_1Sum*gradients_1/loss/log_loss/Mul_1_grad/Mul_1<gradients_1/loss/log_loss/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients_1/loss/log_loss/Mul_1_grad/Reshape_1Reshape*gradients_1/loss/log_loss/Mul_1_grad/Sum_1,gradients_1/loss/log_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients_1/loss/log_loss/Mul_1_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/Mul_1_grad/Reshape/^gradients_1/loss/log_loss/Mul_1_grad/Reshape_1
?
=gradients_1/loss/log_loss/Mul_1_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/Mul_1_grad/Reshape6^gradients_1/loss/log_loss/Mul_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/Mul_1_grad/Reshape*'
_output_shapes
:?????????
?
?gradients_1/loss/log_loss/Mul_1_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/Mul_1_grad/Reshape_16^gradients_1/loss/log_loss/Mul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/Mul_1_grad/Reshape_1*'
_output_shapes
:?????????
m
(gradients_1/loss/log_loss/Mul_grad/ShapeShapelabel*
T0*
out_type0*
_output_shapes
:
{
*gradients_1/loss/log_loss/Mul_grad/Shape_1Shapeloss/log_loss/Log*
T0*
out_type0*
_output_shapes
:
?
8gradients_1/loss/log_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/loss/log_loss/Mul_grad/Shape*gradients_1/loss/log_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients_1/loss/log_loss/Mul_grad/MulMul&gradients_1/loss/log_loss/Neg_grad/Negloss/log_loss/Log*
T0*'
_output_shapes
:?????????
?
&gradients_1/loss/log_loss/Mul_grad/SumSum&gradients_1/loss/log_loss/Mul_grad/Mul8gradients_1/loss/log_loss/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients_1/loss/log_loss/Mul_grad/ReshapeReshape&gradients_1/loss/log_loss/Mul_grad/Sum(gradients_1/loss/log_loss/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/Mul_grad/Mul_1Mullabel&gradients_1/loss/log_loss/Neg_grad/Neg*
T0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/Mul_grad/Sum_1Sum(gradients_1/loss/log_loss/Mul_grad/Mul_1:gradients_1/loss/log_loss/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/Mul_grad/Reshape_1Reshape(gradients_1/loss/log_loss/Mul_grad/Sum_1*gradients_1/loss/log_loss/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
3gradients_1/loss/log_loss/Mul_grad/tuple/group_depsNoOp+^gradients_1/loss/log_loss/Mul_grad/Reshape-^gradients_1/loss/log_loss/Mul_grad/Reshape_1
?
;gradients_1/loss/log_loss/Mul_grad/tuple/control_dependencyIdentity*gradients_1/loss/log_loss/Mul_grad/Reshape4^gradients_1/loss/log_loss/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/loss/log_loss/Mul_grad/Reshape*'
_output_shapes
:?????????
?
=gradients_1/loss/log_loss/Mul_grad/tuple/control_dependency_1Identity,gradients_1/loss/log_loss/Mul_grad/Reshape_14^gradients_1/loss/log_loss/Mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/Mul_grad/Reshape_1*'
_output_shapes
:?????????
?
/gradients_1/loss/log_loss/Log_1_grad/Reciprocal
Reciprocalloss/log_loss/add_1@^gradients_1/loss/log_loss/Mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/Log_1_grad/mulMul?gradients_1/loss/log_loss/Mul_1_grad/tuple/control_dependency_1/gradients_1/loss/log_loss/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
-gradients_1/loss/log_loss/Log_grad/Reciprocal
Reciprocalloss/log_loss/add>^gradients_1/loss/log_loss/Mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
&gradients_1/loss/log_loss/Log_grad/mulMul=gradients_1/loss/log_loss/Mul_grad/tuple/control_dependency_1-gradients_1/loss/log_loss/Log_grad/Reciprocal*
T0*'
_output_shapes
:?????????
}
*gradients_1/loss/log_loss/add_1_grad/ShapeShapeloss/log_loss/sub_1*
T0*
out_type0*
_output_shapes
:
o
,gradients_1/loss/log_loss/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
:gradients_1/loss/log_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/add_1_grad/Shape,gradients_1/loss/log_loss/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients_1/loss/log_loss/add_1_grad/SumSum(gradients_1/loss/log_loss/Log_1_grad/mul:gradients_1/loss/log_loss/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/add_1_grad/ReshapeReshape(gradients_1/loss/log_loss/add_1_grad/Sum*gradients_1/loss/log_loss/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
*gradients_1/loss/log_loss/add_1_grad/Sum_1Sum(gradients_1/loss/log_loss/Log_1_grad/mul<gradients_1/loss/log_loss/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
.gradients_1/loss/log_loss/add_1_grad/Reshape_1Reshape*gradients_1/loss/log_loss/add_1_grad/Sum_1,gradients_1/loss/log_loss/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
5gradients_1/loss/log_loss/add_1_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/add_1_grad/Reshape/^gradients_1/loss/log_loss/add_1_grad/Reshape_1
?
=gradients_1/loss/log_loss/add_1_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/add_1_grad/Reshape6^gradients_1/loss/log_loss/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/add_1_grad/Reshape*'
_output_shapes
:?????????
?
?gradients_1/loss/log_loss/add_1_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/add_1_grad/Reshape_16^gradients_1/loss/log_loss/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/add_1_grad/Reshape_1*
_output_shapes
: 
s
(gradients_1/loss/log_loss/add_grad/ShapeShapesigmoid_out*
T0*
out_type0*
_output_shapes
:
m
*gradients_1/loss/log_loss/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
8gradients_1/loss/log_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/loss/log_loss/add_grad/Shape*gradients_1/loss/log_loss/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
&gradients_1/loss/log_loss/add_grad/SumSum&gradients_1/loss/log_loss/Log_grad/mul8gradients_1/loss/log_loss/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients_1/loss/log_loss/add_grad/ReshapeReshape&gradients_1/loss/log_loss/add_grad/Sum(gradients_1/loss/log_loss/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
(gradients_1/loss/log_loss/add_grad/Sum_1Sum&gradients_1/loss/log_loss/Log_grad/mul:gradients_1/loss/log_loss/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/add_grad/Reshape_1Reshape(gradients_1/loss/log_loss/add_grad/Sum_1*gradients_1/loss/log_loss/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
3gradients_1/loss/log_loss/add_grad/tuple/group_depsNoOp+^gradients_1/loss/log_loss/add_grad/Reshape-^gradients_1/loss/log_loss/add_grad/Reshape_1
?
;gradients_1/loss/log_loss/add_grad/tuple/control_dependencyIdentity*gradients_1/loss/log_loss/add_grad/Reshape4^gradients_1/loss/log_loss/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/loss/log_loss/add_grad/Reshape*'
_output_shapes
:?????????
?
=gradients_1/loss/log_loss/add_grad/tuple/control_dependency_1Identity,gradients_1/loss/log_loss/add_grad/Reshape_14^gradients_1/loss/log_loss/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/add_grad/Reshape_1*
_output_shapes
: 
m
*gradients_1/loss/log_loss/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
w
,gradients_1/loss/log_loss/sub_1_grad/Shape_1Shapesigmoid_out*
T0*
out_type0*
_output_shapes
:
?
:gradients_1/loss/log_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/loss/log_loss/sub_1_grad/Shape,gradients_1/loss/log_loss/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
(gradients_1/loss/log_loss/sub_1_grad/SumSum=gradients_1/loss/log_loss/add_1_grad/tuple/control_dependency:gradients_1/loss/log_loss/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
,gradients_1/loss/log_loss/sub_1_grad/ReshapeReshape(gradients_1/loss/log_loss/sub_1_grad/Sum*gradients_1/loss/log_loss/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
*gradients_1/loss/log_loss/sub_1_grad/Sum_1Sum=gradients_1/loss/log_loss/add_1_grad/tuple/control_dependency<gradients_1/loss/log_loss/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
~
(gradients_1/loss/log_loss/sub_1_grad/NegNeg*gradients_1/loss/log_loss/sub_1_grad/Sum_1*
T0*
_output_shapes
:
?
.gradients_1/loss/log_loss/sub_1_grad/Reshape_1Reshape(gradients_1/loss/log_loss/sub_1_grad/Neg,gradients_1/loss/log_loss/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
5gradients_1/loss/log_loss/sub_1_grad/tuple/group_depsNoOp-^gradients_1/loss/log_loss/sub_1_grad/Reshape/^gradients_1/loss/log_loss/sub_1_grad/Reshape_1
?
=gradients_1/loss/log_loss/sub_1_grad/tuple/control_dependencyIdentity,gradients_1/loss/log_loss/sub_1_grad/Reshape6^gradients_1/loss/log_loss/sub_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/loss/log_loss/sub_1_grad/Reshape*
_output_shapes
: 
?
?gradients_1/loss/log_loss/sub_1_grad/tuple/control_dependency_1Identity.gradients_1/loss/log_loss/sub_1_grad/Reshape_16^gradients_1/loss/log_loss/sub_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/loss/log_loss/sub_1_grad/Reshape_1*'
_output_shapes
:?????????
?
gradients_1/AddNAddN;gradients_1/loss/log_loss/add_grad/tuple/control_dependency?gradients_1/loss/log_loss/sub_1_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@gradients_1/loss/log_loss/add_grad/Reshape*
N*'
_output_shapes
:?????????
?
(gradients_1/sigmoid_out_grad/SigmoidGradSigmoidGradsigmoid_outgradients_1/AddN*
T0*'
_output_shapes
:?????????
k
!gradients_1/output/Add_grad/ShapeShape
output/Mul*
T0*
out_type0*
_output_shapes
:
f
#gradients_1/output/Add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
1gradients_1/output/Add_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_1/output/Add_grad/Shape#gradients_1/output/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/output/Add_grad/SumSum(gradients_1/sigmoid_out_grad/SigmoidGrad1gradients_1/output/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
#gradients_1/output/Add_grad/ReshapeReshapegradients_1/output/Add_grad/Sum!gradients_1/output/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
!gradients_1/output/Add_grad/Sum_1Sum(gradients_1/sigmoid_out_grad/SigmoidGrad3gradients_1/output/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
%gradients_1/output/Add_grad/Reshape_1Reshape!gradients_1/output/Add_grad/Sum_1#gradients_1/output/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
,gradients_1/output/Add_grad/tuple/group_depsNoOp$^gradients_1/output/Add_grad/Reshape&^gradients_1/output/Add_grad/Reshape_1
?
4gradients_1/output/Add_grad/tuple/control_dependencyIdentity#gradients_1/output/Add_grad/Reshape-^gradients_1/output/Add_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/output/Add_grad/Reshape*'
_output_shapes
:?????????
?
6gradients_1/output/Add_grad/tuple/control_dependency_1Identity%gradients_1/output/Add_grad/Reshape_1-^gradients_1/output/Add_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients_1/output/Add_grad/Reshape_1*
_output_shapes
: 
?
"gradients_1/output/Mul_grad/MatMulMatMul4gradients_1/output/Add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????<
?
$gradients_1/output/Mul_grad/MatMul_1MatMuldeep_fm/concat4gradients_1/output/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:<
?
,gradients_1/output/Mul_grad/tuple/group_depsNoOp#^gradients_1/output/Mul_grad/MatMul%^gradients_1/output/Mul_grad/MatMul_1
?
4gradients_1/output/Mul_grad/tuple/control_dependencyIdentity"gradients_1/output/Mul_grad/MatMul-^gradients_1/output/Mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/output/Mul_grad/MatMul*'
_output_shapes
:?????????<
?
6gradients_1/output/Mul_grad/tuple/control_dependency_1Identity$gradients_1/output/Mul_grad/MatMul_1-^gradients_1/output/Mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/output/Mul_grad/MatMul_1*
_output_shapes

:<
f
$gradients_1/deep_fm/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
#gradients_1/deep_fm/concat_grad/modFloorModdeep_fm/concat/axis$gradients_1/deep_fm/concat_grad/Rank*
T0*
_output_shapes
: 
~
%gradients_1/deep_fm/concat_grad/ShapeShapefirst_order/dropout/mul_1*
T0*
out_type0*
_output_shapes
:
?
&gradients_1/deep_fm/concat_grad/ShapeNShapeNfirst_order/dropout/mul_1dropout/mul_1deep_component/dropout_2/mul_1*
T0*
out_type0*
N*&
_output_shapes
:::
?
,gradients_1/deep_fm/concat_grad/ConcatOffsetConcatOffset#gradients_1/deep_fm/concat_grad/mod&gradients_1/deep_fm/concat_grad/ShapeN(gradients_1/deep_fm/concat_grad/ShapeN:1(gradients_1/deep_fm/concat_grad/ShapeN:2*
N*&
_output_shapes
:::
?
%gradients_1/deep_fm/concat_grad/SliceSlice4gradients_1/output/Mul_grad/tuple/control_dependency,gradients_1/deep_fm/concat_grad/ConcatOffset&gradients_1/deep_fm/concat_grad/ShapeN*
T0*
Index0*'
_output_shapes
:?????????
?
'gradients_1/deep_fm/concat_grad/Slice_1Slice4gradients_1/output/Mul_grad/tuple/control_dependency.gradients_1/deep_fm/concat_grad/ConcatOffset:1(gradients_1/deep_fm/concat_grad/ShapeN:1*
T0*
Index0*'
_output_shapes
:?????????
?
'gradients_1/deep_fm/concat_grad/Slice_2Slice4gradients_1/output/Mul_grad/tuple/control_dependency.gradients_1/deep_fm/concat_grad/ConcatOffset:2(gradients_1/deep_fm/concat_grad/ShapeN:2*
T0*
Index0*'
_output_shapes
:????????? 
?
0gradients_1/deep_fm/concat_grad/tuple/group_depsNoOp&^gradients_1/deep_fm/concat_grad/Slice(^gradients_1/deep_fm/concat_grad/Slice_1(^gradients_1/deep_fm/concat_grad/Slice_2
?
8gradients_1/deep_fm/concat_grad/tuple/control_dependencyIdentity%gradients_1/deep_fm/concat_grad/Slice1^gradients_1/deep_fm/concat_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients_1/deep_fm/concat_grad/Slice*'
_output_shapes
:?????????
?
:gradients_1/deep_fm/concat_grad/tuple/control_dependency_1Identity'gradients_1/deep_fm/concat_grad/Slice_11^gradients_1/deep_fm/concat_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/deep_fm/concat_grad/Slice_1*'
_output_shapes
:?????????
?
:gradients_1/deep_fm/concat_grad/tuple/control_dependency_2Identity'gradients_1/deep_fm/concat_grad/Slice_21^gradients_1/deep_fm/concat_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_1/deep_fm/concat_grad/Slice_2*'
_output_shapes
:????????? 
?
0gradients_1/first_order/dropout/mul_1_grad/ShapeShapefirst_order/dropout/mul*
T0*
out_type0*
_output_shapes
:
?
2gradients_1/first_order/dropout/mul_1_grad/Shape_1Shapefirst_order/dropout/Cast*
T0*
out_type0*
_output_shapes
:
?
@gradients_1/first_order/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/first_order/dropout/mul_1_grad/Shape2gradients_1/first_order/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
.gradients_1/first_order/dropout/mul_1_grad/MulMul8gradients_1/deep_fm/concat_grad/tuple/control_dependencyfirst_order/dropout/Cast*
T0*'
_output_shapes
:?????????
?
.gradients_1/first_order/dropout/mul_1_grad/SumSum.gradients_1/first_order/dropout/mul_1_grad/Mul@gradients_1/first_order/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
2gradients_1/first_order/dropout/mul_1_grad/ReshapeReshape.gradients_1/first_order/dropout/mul_1_grad/Sum0gradients_1/first_order/dropout/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
0gradients_1/first_order/dropout/mul_1_grad/Mul_1Mulfirst_order/dropout/mul8gradients_1/deep_fm/concat_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
0gradients_1/first_order/dropout/mul_1_grad/Sum_1Sum0gradients_1/first_order/dropout/mul_1_grad/Mul_1Bgradients_1/first_order/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
4gradients_1/first_order/dropout/mul_1_grad/Reshape_1Reshape0gradients_1/first_order/dropout/mul_1_grad/Sum_12gradients_1/first_order/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
;gradients_1/first_order/dropout/mul_1_grad/tuple/group_depsNoOp3^gradients_1/first_order/dropout/mul_1_grad/Reshape5^gradients_1/first_order/dropout/mul_1_grad/Reshape_1
?
Cgradients_1/first_order/dropout/mul_1_grad/tuple/control_dependencyIdentity2gradients_1/first_order/dropout/mul_1_grad/Reshape<^gradients_1/first_order/dropout/mul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/first_order/dropout/mul_1_grad/Reshape*'
_output_shapes
:?????????
?
Egradients_1/first_order/dropout/mul_1_grad/tuple/control_dependency_1Identity4gradients_1/first_order/dropout/mul_1_grad/Reshape_1<^gradients_1/first_order/dropout/mul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/first_order/dropout/mul_1_grad/Reshape_1*'
_output_shapes
:?????????
o
$gradients_1/dropout/mul_1_grad/ShapeShapedropout/mul*
T0*
out_type0*
_output_shapes
:
r
&gradients_1/dropout/mul_1_grad/Shape_1Shapedropout/Cast*
T0*
out_type0*
_output_shapes
:
?
4gradients_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/dropout/mul_1_grad/Shape&gradients_1/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
"gradients_1/dropout/mul_1_grad/MulMul:gradients_1/deep_fm/concat_grad/tuple/control_dependency_1dropout/Cast*
T0*'
_output_shapes
:?????????
?
"gradients_1/dropout/mul_1_grad/SumSum"gradients_1/dropout/mul_1_grad/Mul4gradients_1/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
&gradients_1/dropout/mul_1_grad/ReshapeReshape"gradients_1/dropout/mul_1_grad/Sum$gradients_1/dropout/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
$gradients_1/dropout/mul_1_grad/Mul_1Muldropout/mul:gradients_1/deep_fm/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
$gradients_1/dropout/mul_1_grad/Sum_1Sum$gradients_1/dropout/mul_1_grad/Mul_16gradients_1/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients_1/dropout/mul_1_grad/Reshape_1Reshape$gradients_1/dropout/mul_1_grad/Sum_1&gradients_1/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
/gradients_1/dropout/mul_1_grad/tuple/group_depsNoOp'^gradients_1/dropout/mul_1_grad/Reshape)^gradients_1/dropout/mul_1_grad/Reshape_1
?
7gradients_1/dropout/mul_1_grad/tuple/control_dependencyIdentity&gradients_1/dropout/mul_1_grad/Reshape0^gradients_1/dropout/mul_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/dropout/mul_1_grad/Reshape*'
_output_shapes
:?????????
?
9gradients_1/dropout/mul_1_grad/tuple/control_dependency_1Identity(gradients_1/dropout/mul_1_grad/Reshape_10^gradients_1/dropout/mul_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/dropout/mul_1_grad/Reshape_1*'
_output_shapes
:?????????
?
5gradients_1/deep_component/dropout_2/mul_1_grad/ShapeShapedeep_component/dropout_2/mul*
T0*
out_type0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_2/mul_1_grad/Shape_1Shapedeep_component/dropout_2/Cast*
T0*
out_type0*
_output_shapes
:
?
Egradients_1/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/deep_component/dropout_2/mul_1_grad/Shape7gradients_1/deep_component/dropout_2/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
3gradients_1/deep_component/dropout_2/mul_1_grad/MulMul:gradients_1/deep_fm/concat_grad/tuple/control_dependency_2deep_component/dropout_2/Cast*
T0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_2/mul_1_grad/SumSum3gradients_1/deep_component/dropout_2/mul_1_grad/MulEgradients_1/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_2/mul_1_grad/ReshapeReshape3gradients_1/deep_component/dropout_2/mul_1_grad/Sum5gradients_1/deep_component/dropout_2/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
5gradients_1/deep_component/dropout_2/mul_1_grad/Mul_1Muldeep_component/dropout_2/mul:gradients_1/deep_fm/concat_grad/tuple/control_dependency_2*
T0*'
_output_shapes
:????????? 
?
5gradients_1/deep_component/dropout_2/mul_1_grad/Sum_1Sum5gradients_1/deep_component/dropout_2/mul_1_grad/Mul_1Ggradients_1/deep_component/dropout_2/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
9gradients_1/deep_component/dropout_2/mul_1_grad/Reshape_1Reshape5gradients_1/deep_component/dropout_2/mul_1_grad/Sum_17gradients_1/deep_component/dropout_2/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:????????? 
?
@gradients_1/deep_component/dropout_2/mul_1_grad/tuple/group_depsNoOp8^gradients_1/deep_component/dropout_2/mul_1_grad/Reshape:^gradients_1/deep_component/dropout_2/mul_1_grad/Reshape_1
?
Hgradients_1/deep_component/dropout_2/mul_1_grad/tuple/control_dependencyIdentity7gradients_1/deep_component/dropout_2/mul_1_grad/ReshapeA^gradients_1/deep_component/dropout_2/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/deep_component/dropout_2/mul_1_grad/Reshape*'
_output_shapes
:????????? 
?
Jgradients_1/deep_component/dropout_2/mul_1_grad/tuple/control_dependency_1Identity9gradients_1/deep_component/dropout_2/mul_1_grad/Reshape_1A^gradients_1/deep_component/dropout_2/mul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/deep_component/dropout_2/mul_1_grad/Reshape_1*'
_output_shapes
:????????? 
}
.gradients_1/first_order/dropout/mul_grad/ShapeShapefirst_order/Sum*
T0*
out_type0*
_output_shapes
:
s
0gradients_1/first_order/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
>gradients_1/first_order/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_1/first_order/dropout/mul_grad/Shape0gradients_1/first_order/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
,gradients_1/first_order/dropout/mul_grad/MulMulCgradients_1/first_order/dropout/mul_1_grad/tuple/control_dependencyfirst_order/dropout/truediv*
T0*'
_output_shapes
:?????????
?
,gradients_1/first_order/dropout/mul_grad/SumSum,gradients_1/first_order/dropout/mul_grad/Mul>gradients_1/first_order/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
0gradients_1/first_order/dropout/mul_grad/ReshapeReshape,gradients_1/first_order/dropout/mul_grad/Sum.gradients_1/first_order/dropout/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
.gradients_1/first_order/dropout/mul_grad/Mul_1Mulfirst_order/SumCgradients_1/first_order/dropout/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
.gradients_1/first_order/dropout/mul_grad/Sum_1Sum.gradients_1/first_order/dropout/mul_grad/Mul_1@gradients_1/first_order/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
2gradients_1/first_order/dropout/mul_grad/Reshape_1Reshape.gradients_1/first_order/dropout/mul_grad/Sum_10gradients_1/first_order/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
9gradients_1/first_order/dropout/mul_grad/tuple/group_depsNoOp1^gradients_1/first_order/dropout/mul_grad/Reshape3^gradients_1/first_order/dropout/mul_grad/Reshape_1
?
Agradients_1/first_order/dropout/mul_grad/tuple/control_dependencyIdentity0gradients_1/first_order/dropout/mul_grad/Reshape:^gradients_1/first_order/dropout/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/first_order/dropout/mul_grad/Reshape*'
_output_shapes
:?????????
?
Cgradients_1/first_order/dropout/mul_grad/tuple/control_dependency_1Identity2gradients_1/first_order/dropout/mul_grad/Reshape_1:^gradients_1/first_order/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/first_order/dropout/mul_grad/Reshape_1*
_output_shapes
: 
e
"gradients_1/dropout/mul_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
g
$gradients_1/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
2gradients_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/dropout/mul_grad/Shape$gradients_1/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
 gradients_1/dropout/mul_grad/MulMul7gradients_1/dropout/mul_1_grad/tuple/control_dependencydropout/truediv*
T0*'
_output_shapes
:?????????
?
 gradients_1/dropout/mul_grad/SumSum gradients_1/dropout/mul_grad/Mul2gradients_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
$gradients_1/dropout/mul_grad/ReshapeReshape gradients_1/dropout/mul_grad/Sum"gradients_1/dropout/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
"gradients_1/dropout/mul_grad/Mul_1Mulmul7gradients_1/dropout/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
"gradients_1/dropout/mul_grad/Sum_1Sum"gradients_1/dropout/mul_grad/Mul_14gradients_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
&gradients_1/dropout/mul_grad/Reshape_1Reshape"gradients_1/dropout/mul_grad/Sum_1$gradients_1/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
-gradients_1/dropout/mul_grad/tuple/group_depsNoOp%^gradients_1/dropout/mul_grad/Reshape'^gradients_1/dropout/mul_grad/Reshape_1
?
5gradients_1/dropout/mul_grad/tuple/control_dependencyIdentity$gradients_1/dropout/mul_grad/Reshape.^gradients_1/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/dropout/mul_grad/Reshape*'
_output_shapes
:?????????
?
7gradients_1/dropout/mul_grad/tuple/control_dependency_1Identity&gradients_1/dropout/mul_grad/Reshape_1.^gradients_1/dropout/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/dropout/mul_grad/Reshape_1*
_output_shapes
: 
?
3gradients_1/deep_component/dropout_2/mul_grad/ShapeShapedeep_component/Relu_1*
T0*
out_type0*
_output_shapes
:
x
5gradients_1/deep_component/dropout_2/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Cgradients_1/deep_component/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/deep_component/dropout_2/mul_grad/Shape5gradients_1/deep_component/dropout_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients_1/deep_component/dropout_2/mul_grad/MulMulHgradients_1/deep_component/dropout_2/mul_1_grad/tuple/control_dependency deep_component/dropout_2/truediv*
T0*'
_output_shapes
:????????? 
?
1gradients_1/deep_component/dropout_2/mul_grad/SumSum1gradients_1/deep_component/dropout_2/mul_grad/MulCgradients_1/deep_component/dropout_2/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients_1/deep_component/dropout_2/mul_grad/ReshapeReshape1gradients_1/deep_component/dropout_2/mul_grad/Sum3gradients_1/deep_component/dropout_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_2/mul_grad/Mul_1Muldeep_component/Relu_1Hgradients_1/deep_component/dropout_2/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_2/mul_grad/Sum_1Sum3gradients_1/deep_component/dropout_2/mul_grad/Mul_1Egradients_1/deep_component/dropout_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_2/mul_grad/Reshape_1Reshape3gradients_1/deep_component/dropout_2/mul_grad/Sum_15gradients_1/deep_component/dropout_2/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
>gradients_1/deep_component/dropout_2/mul_grad/tuple/group_depsNoOp6^gradients_1/deep_component/dropout_2/mul_grad/Reshape8^gradients_1/deep_component/dropout_2/mul_grad/Reshape_1
?
Fgradients_1/deep_component/dropout_2/mul_grad/tuple/control_dependencyIdentity5gradients_1/deep_component/dropout_2/mul_grad/Reshape?^gradients_1/deep_component/dropout_2/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/deep_component/dropout_2/mul_grad/Reshape*'
_output_shapes
:????????? 
?
Hgradients_1/deep_component/dropout_2/mul_grad/tuple/control_dependency_1Identity7gradients_1/deep_component/dropout_2/mul_grad/Reshape_1?^gradients_1/deep_component/dropout_2/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/deep_component/dropout_2/mul_grad/Reshape_1*
_output_shapes
: 
u
&gradients_1/first_order/Sum_grad/ShapeShapefirst_order/Mul*
T0*
out_type0*
_output_shapes
:
?
%gradients_1/first_order/Sum_grad/SizeConst*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
$gradients_1/first_order/Sum_grad/addAdd!first_order/Sum/reduction_indices%gradients_1/first_order/Sum_grad/Size*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
_output_shapes
: 
?
$gradients_1/first_order/Sum_grad/modFloorMod$gradients_1/first_order/Sum_grad/add%gradients_1/first_order/Sum_grad/Size*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
_output_shapes
: 
?
(gradients_1/first_order/Sum_grad/Shape_1Const*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
,gradients_1/first_order/Sum_grad/range/startConst*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
,gradients_1/first_order/Sum_grad/range/deltaConst*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
&gradients_1/first_order/Sum_grad/rangeRange,gradients_1/first_order/Sum_grad/range/start%gradients_1/first_order/Sum_grad/Size,gradients_1/first_order/Sum_grad/range/delta*

Tidx0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
_output_shapes
:
?
+gradients_1/first_order/Sum_grad/Fill/valueConst*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients_1/first_order/Sum_grad/FillFill(gradients_1/first_order/Sum_grad/Shape_1+gradients_1/first_order/Sum_grad/Fill/value*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
.gradients_1/first_order/Sum_grad/DynamicStitchDynamicStitch&gradients_1/first_order/Sum_grad/range$gradients_1/first_order/Sum_grad/mod&gradients_1/first_order/Sum_grad/Shape%gradients_1/first_order/Sum_grad/Fill*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
N*
_output_shapes
:
?
*gradients_1/first_order/Sum_grad/Maximum/yConst*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
(gradients_1/first_order/Sum_grad/MaximumMaximum.gradients_1/first_order/Sum_grad/DynamicStitch*gradients_1/first_order/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
_output_shapes
:
?
)gradients_1/first_order/Sum_grad/floordivFloorDiv&gradients_1/first_order/Sum_grad/Shape(gradients_1/first_order/Sum_grad/Maximum*
T0*9
_class/
-+loc:@gradients_1/first_order/Sum_grad/Shape*
_output_shapes
:
?
(gradients_1/first_order/Sum_grad/ReshapeReshapeAgradients_1/first_order/dropout/mul_grad/tuple/control_dependency.gradients_1/first_order/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
%gradients_1/first_order/Sum_grad/TileTile(gradients_1/first_order/Sum_grad/Reshape)gradients_1/first_order/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
]
gradients_1/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
gradients_1/mul_grad/Shape_1ShapeSub*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_grad/Shapegradients_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/mul_grad/MulMul5gradients_1/dropout/mul_grad/tuple/control_dependencySub*
T0*'
_output_shapes
:?????????
?
gradients_1/mul_grad/SumSumgradients_1/mul_grad/Mul*gradients_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients_1/mul_grad/ReshapeReshapegradients_1/mul_grad/Sumgradients_1/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
gradients_1/mul_grad/Mul_1Mulmul/x5gradients_1/dropout/mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
gradients_1/mul_grad/Sum_1Sumgradients_1/mul_grad/Mul_1,gradients_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients_1/mul_grad/Reshape_1Reshapegradients_1/mul_grad/Sum_1gradients_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
m
%gradients_1/mul_grad/tuple/group_depsNoOp^gradients_1/mul_grad/Reshape^gradients_1/mul_grad/Reshape_1
?
-gradients_1/mul_grad/tuple/control_dependencyIdentitygradients_1/mul_grad/Reshape&^gradients_1/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_grad/Reshape*
_output_shapes
: 
?
/gradients_1/mul_grad/tuple/control_dependency_1Identitygradients_1/mul_grad/Reshape_1&^gradients_1/mul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_grad/Reshape_1*'
_output_shapes
:?????????
?
/gradients_1/deep_component/Relu_1_grad/ReluGradReluGradFgradients_1/deep_component/dropout_2/mul_grad/tuple/control_dependencydeep_component/Relu_1*
T0*'
_output_shapes
:????????? 
?
&gradients_1/first_order/Mul_grad/ShapeShape%first_order/embedding_lookup/Identity*
T0*
out_type0*
_output_shapes
:
?
(gradients_1/first_order/Mul_grad/Shape_1Shapeembedding_lookup/Reshape*
T0*
out_type0*
_output_shapes
:
?
6gradients_1/first_order/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/first_order/Mul_grad/Shape(gradients_1/first_order/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
$gradients_1/first_order/Mul_grad/MulMul%gradients_1/first_order/Sum_grad/Tileembedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
?
$gradients_1/first_order/Mul_grad/SumSum$gradients_1/first_order/Mul_grad/Mul6gradients_1/first_order/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
(gradients_1/first_order/Mul_grad/ReshapeReshape$gradients_1/first_order/Mul_grad/Sum&gradients_1/first_order/Mul_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
&gradients_1/first_order/Mul_grad/Mul_1Mul%first_order/embedding_lookup/Identity%gradients_1/first_order/Sum_grad/Tile*
T0*+
_output_shapes
:?????????
?
&gradients_1/first_order/Mul_grad/Sum_1Sum&gradients_1/first_order/Mul_grad/Mul_18gradients_1/first_order/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
*gradients_1/first_order/Mul_grad/Reshape_1Reshape&gradients_1/first_order/Mul_grad/Sum_1(gradients_1/first_order/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
1gradients_1/first_order/Mul_grad/tuple/group_depsNoOp)^gradients_1/first_order/Mul_grad/Reshape+^gradients_1/first_order/Mul_grad/Reshape_1
?
9gradients_1/first_order/Mul_grad/tuple/control_dependencyIdentity(gradients_1/first_order/Mul_grad/Reshape2^gradients_1/first_order/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/first_order/Mul_grad/Reshape*4
_output_shapes"
 :??????????????????
?
;gradients_1/first_order/Mul_grad/tuple/control_dependency_1Identity*gradients_1/first_order/Mul_grad/Reshape_12^gradients_1/first_order/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/first_order/Mul_grad/Reshape_1*+
_output_shapes
:?????????
m
gradients_1/Sub_grad/ShapeShapesecond_order/Square*
T0*
out_type0*
_output_shapes
:
_
gradients_1/Sub_grad/Shape_1ShapeSum*
T0*
out_type0*
_output_shapes
:
?
*gradients_1/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_grad/Shapegradients_1/Sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/Sub_grad/SumSum/gradients_1/mul_grad/tuple/control_dependency_1*gradients_1/Sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients_1/Sub_grad/ReshapeReshapegradients_1/Sub_grad/Sumgradients_1/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/Sub_grad/Sum_1Sum/gradients_1/mul_grad/tuple/control_dependency_1,gradients_1/Sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
^
gradients_1/Sub_grad/NegNeggradients_1/Sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients_1/Sub_grad/Reshape_1Reshapegradients_1/Sub_grad/Neggradients_1/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
m
%gradients_1/Sub_grad/tuple/group_depsNoOp^gradients_1/Sub_grad/Reshape^gradients_1/Sub_grad/Reshape_1
?
-gradients_1/Sub_grad/tuple/control_dependencyIdentitygradients_1/Sub_grad/Reshape&^gradients_1/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Sub_grad/Reshape*'
_output_shapes
:?????????
?
/gradients_1/Sub_grad/tuple/control_dependency_1Identitygradients_1/Sub_grad/Reshape_1&^gradients_1/Sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Sub_grad/Reshape_1*'
_output_shapes
:?????????
?
+gradients_1/deep_component/Add_1_grad/ShapeShapedeep_component/MatMul_1*
T0*
out_type0*
_output_shapes
:
~
-gradients_1/deep_component/Add_1_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
;gradients_1/deep_component/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients_1/deep_component/Add_1_grad/Shape-gradients_1/deep_component/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
)gradients_1/deep_component/Add_1_grad/SumSum/gradients_1/deep_component/Relu_1_grad/ReluGrad;gradients_1/deep_component/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
-gradients_1/deep_component/Add_1_grad/ReshapeReshape)gradients_1/deep_component/Add_1_grad/Sum+gradients_1/deep_component/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
+gradients_1/deep_component/Add_1_grad/Sum_1Sum/gradients_1/deep_component/Relu_1_grad/ReluGrad=gradients_1/deep_component/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
/gradients_1/deep_component/Add_1_grad/Reshape_1Reshape+gradients_1/deep_component/Add_1_grad/Sum_1-gradients_1/deep_component/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
?
6gradients_1/deep_component/Add_1_grad/tuple/group_depsNoOp.^gradients_1/deep_component/Add_1_grad/Reshape0^gradients_1/deep_component/Add_1_grad/Reshape_1
?
>gradients_1/deep_component/Add_1_grad/tuple/control_dependencyIdentity-gradients_1/deep_component/Add_1_grad/Reshape7^gradients_1/deep_component/Add_1_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/deep_component/Add_1_grad/Reshape*'
_output_shapes
:????????? 
?
@gradients_1/deep_component/Add_1_grad/tuple/control_dependency_1Identity/gradients_1/deep_component/Add_1_grad/Reshape_17^gradients_1/deep_component/Add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/deep_component/Add_1_grad/Reshape_1*
_output_shapes

: 
?
*gradients_1/second_order/Square_grad/ConstConst.^gradients_1/Sub_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
(gradients_1/second_order/Square_grad/MulMulsecond_order/Sum*gradients_1/second_order/Square_grad/Const*
T0*'
_output_shapes
:?????????
?
*gradients_1/second_order/Square_grad/Mul_1Mul-gradients_1/Sub_grad/tuple/control_dependency(gradients_1/second_order/Square_grad/Mul*
T0*'
_output_shapes
:?????????
`
gradients_1/Sum_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
?
gradients_1/Sum_grad/SizeConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients_1/Sum_grad/addAddSum/reduction_indicesgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
_output_shapes
: 
?
gradients_1/Sum_grad/modFloorModgradients_1/Sum_grad/addgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
_output_shapes
: 
?
gradients_1/Sum_grad/Shape_1Const*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
 gradients_1/Sum_grad/range/startConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
 gradients_1/Sum_grad/range/deltaConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients_1/Sum_grad/rangeRange gradients_1/Sum_grad/range/startgradients_1/Sum_grad/Size gradients_1/Sum_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
_output_shapes
:
?
gradients_1/Sum_grad/Fill/valueConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients_1/Sum_grad/FillFillgradients_1/Sum_grad/Shape_1gradients_1/Sum_grad/Fill/value*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
"gradients_1/Sum_grad/DynamicStitchDynamicStitchgradients_1/Sum_grad/rangegradients_1/Sum_grad/modgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Fill*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
N*
_output_shapes
:
?
gradients_1/Sum_grad/Maximum/yConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients_1/Sum_grad/MaximumMaximum"gradients_1/Sum_grad/DynamicStitchgradients_1/Sum_grad/Maximum/y*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
_output_shapes
:
?
gradients_1/Sum_grad/floordivFloorDivgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Maximum*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
_output_shapes
:
?
gradients_1/Sum_grad/ReshapeReshape/gradients_1/Sub_grad/tuple/control_dependency_1"gradients_1/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
gradients_1/Sum_grad/TileTilegradients_1/Sum_grad/Reshapegradients_1/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
?
/gradients_1/deep_component/MatMul_1_grad/MatMulMatMul>gradients_1/deep_component/Add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:????????? 
?
1gradients_1/deep_component/MatMul_1_grad/MatMul_1MatMuldeep_component/dropout_1/mul_1>gradients_1/deep_component/Add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:  
?
9gradients_1/deep_component/MatMul_1_grad/tuple/group_depsNoOp0^gradients_1/deep_component/MatMul_1_grad/MatMul2^gradients_1/deep_component/MatMul_1_grad/MatMul_1
?
Agradients_1/deep_component/MatMul_1_grad/tuple/control_dependencyIdentity/gradients_1/deep_component/MatMul_1_grad/MatMul:^gradients_1/deep_component/MatMul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/deep_component/MatMul_1_grad/MatMul*'
_output_shapes
:????????? 
?
Cgradients_1/deep_component/MatMul_1_grad/tuple/control_dependency_1Identity1gradients_1/deep_component/MatMul_1_grad/MatMul_1:^gradients_1/deep_component/MatMul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/deep_component/MatMul_1_grad/MatMul_1*
_output_shapes

:  
?
3gradients_1/first_order/embedding_lookup_grad/ShapeConst*
_class
loc:@feature_bias*%
valueB	"?              *
dtype0	*
_output_shapes
:
?
2gradients_1/first_order/embedding_lookup_grad/CastCast3gradients_1/first_order/embedding_lookup_grad/Shape*

SrcT0	*
_class
loc:@feature_bias*
Truncate( *

DstT0*
_output_shapes
:
w
2gradients_1/first_order/embedding_lookup_grad/SizeSize
feat_index*
T0*
out_type0*
_output_shapes
: 
~
<gradients_1/first_order/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
8gradients_1/first_order/embedding_lookup_grad/ExpandDims
ExpandDims2gradients_1/first_order/embedding_lookup_grad/Size<gradients_1/first_order/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Agradients_1/first_order/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Cgradients_1/first_order/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Cgradients_1/first_order/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
;gradients_1/first_order/embedding_lookup_grad/strided_sliceStridedSlice2gradients_1/first_order/embedding_lookup_grad/CastAgradients_1/first_order/embedding_lookup_grad/strided_slice/stackCgradients_1/first_order/embedding_lookup_grad/strided_slice/stack_1Cgradients_1/first_order/embedding_lookup_grad/strided_slice/stack_2*
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
9gradients_1/first_order/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
4gradients_1/first_order/embedding_lookup_grad/concatConcatV28gradients_1/first_order/embedding_lookup_grad/ExpandDims;gradients_1/first_order/embedding_lookup_grad/strided_slice9gradients_1/first_order/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
5gradients_1/first_order/embedding_lookup_grad/ReshapeReshape9gradients_1/first_order/Mul_grad/tuple/control_dependency4gradients_1/first_order/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
7gradients_1/first_order/embedding_lookup_grad/Reshape_1Reshape
feat_index8gradients_1/first_order/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
{
'gradients_1/second_order/Sum_grad/ShapeShapeembedding_lookup/Mul*
T0*
out_type0*
_output_shapes
:
?
&gradients_1/second_order/Sum_grad/SizeConst*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
%gradients_1/second_order/Sum_grad/addAdd"second_order/Sum/reduction_indices&gradients_1/second_order/Sum_grad/Size*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
_output_shapes
: 
?
%gradients_1/second_order/Sum_grad/modFloorMod%gradients_1/second_order/Sum_grad/add&gradients_1/second_order/Sum_grad/Size*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
_output_shapes
: 
?
)gradients_1/second_order/Sum_grad/Shape_1Const*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
?
-gradients_1/second_order/Sum_grad/range/startConst*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
-gradients_1/second_order/Sum_grad/range/deltaConst*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
'gradients_1/second_order/Sum_grad/rangeRange-gradients_1/second_order/Sum_grad/range/start&gradients_1/second_order/Sum_grad/Size-gradients_1/second_order/Sum_grad/range/delta*

Tidx0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
_output_shapes
:
?
,gradients_1/second_order/Sum_grad/Fill/valueConst*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
&gradients_1/second_order/Sum_grad/FillFill)gradients_1/second_order/Sum_grad/Shape_1,gradients_1/second_order/Sum_grad/Fill/value*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*

index_type0*
_output_shapes
: 
?
/gradients_1/second_order/Sum_grad/DynamicStitchDynamicStitch'gradients_1/second_order/Sum_grad/range%gradients_1/second_order/Sum_grad/mod'gradients_1/second_order/Sum_grad/Shape&gradients_1/second_order/Sum_grad/Fill*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
N*
_output_shapes
:
?
+gradients_1/second_order/Sum_grad/Maximum/yConst*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
)gradients_1/second_order/Sum_grad/MaximumMaximum/gradients_1/second_order/Sum_grad/DynamicStitch+gradients_1/second_order/Sum_grad/Maximum/y*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
_output_shapes
:
?
*gradients_1/second_order/Sum_grad/floordivFloorDiv'gradients_1/second_order/Sum_grad/Shape)gradients_1/second_order/Sum_grad/Maximum*
T0*:
_class0
.,loc:@gradients_1/second_order/Sum_grad/Shape*
_output_shapes
:
?
)gradients_1/second_order/Sum_grad/ReshapeReshape*gradients_1/second_order/Square_grad/Mul_1/gradients_1/second_order/Sum_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'???????????????????????????
?
&gradients_1/second_order/Sum_grad/TileTile)gradients_1/second_order/Sum_grad/Reshape*gradients_1/second_order/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
:?????????
~
gradients_1/Square_grad/ConstConst^gradients_1/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
gradients_1/Square_grad/MulMulembedding_lookup/Mulgradients_1/Square_grad/Const*
T0*+
_output_shapes
:?????????
?
gradients_1/Square_grad/Mul_1Mulgradients_1/Sum_grad/Tilegradients_1/Square_grad/Mul*
T0*+
_output_shapes
:?????????
?
5gradients_1/deep_component/dropout_1/mul_1_grad/ShapeShapedeep_component/dropout_1/mul*
T0*
out_type0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_1/mul_1_grad/Shape_1Shapedeep_component/dropout_1/Cast*
T0*
out_type0*
_output_shapes
:
?
Egradients_1/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/deep_component/dropout_1/mul_1_grad/Shape7gradients_1/deep_component/dropout_1/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
3gradients_1/deep_component/dropout_1/mul_1_grad/MulMulAgradients_1/deep_component/MatMul_1_grad/tuple/control_dependencydeep_component/dropout_1/Cast*
T0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_1/mul_1_grad/SumSum3gradients_1/deep_component/dropout_1/mul_1_grad/MulEgradients_1/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_1/mul_1_grad/ReshapeReshape3gradients_1/deep_component/dropout_1/mul_1_grad/Sum5gradients_1/deep_component/dropout_1/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
5gradients_1/deep_component/dropout_1/mul_1_grad/Mul_1Muldeep_component/dropout_1/mulAgradients_1/deep_component/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
5gradients_1/deep_component/dropout_1/mul_1_grad/Sum_1Sum5gradients_1/deep_component/dropout_1/mul_1_grad/Mul_1Ggradients_1/deep_component/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
9gradients_1/deep_component/dropout_1/mul_1_grad/Reshape_1Reshape5gradients_1/deep_component/dropout_1/mul_1_grad/Sum_17gradients_1/deep_component/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:????????? 
?
@gradients_1/deep_component/dropout_1/mul_1_grad/tuple/group_depsNoOp8^gradients_1/deep_component/dropout_1/mul_1_grad/Reshape:^gradients_1/deep_component/dropout_1/mul_1_grad/Reshape_1
?
Hgradients_1/deep_component/dropout_1/mul_1_grad/tuple/control_dependencyIdentity7gradients_1/deep_component/dropout_1/mul_1_grad/ReshapeA^gradients_1/deep_component/dropout_1/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/deep_component/dropout_1/mul_1_grad/Reshape*'
_output_shapes
:????????? 
?
Jgradients_1/deep_component/dropout_1/mul_1_grad/tuple/control_dependency_1Identity9gradients_1/deep_component/dropout_1/mul_1_grad/Reshape_1A^gradients_1/deep_component/dropout_1/mul_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/deep_component/dropout_1/mul_1_grad/Reshape_1*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_1/mul_grad/ShapeShapedeep_component/Relu*
T0*
out_type0*
_output_shapes
:
x
5gradients_1/deep_component/dropout_1/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Cgradients_1/deep_component/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/deep_component/dropout_1/mul_grad/Shape5gradients_1/deep_component/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients_1/deep_component/dropout_1/mul_grad/MulMulHgradients_1/deep_component/dropout_1/mul_1_grad/tuple/control_dependency deep_component/dropout_1/truediv*
T0*'
_output_shapes
:????????? 
?
1gradients_1/deep_component/dropout_1/mul_grad/SumSum1gradients_1/deep_component/dropout_1/mul_grad/MulCgradients_1/deep_component/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients_1/deep_component/dropout_1/mul_grad/ReshapeReshape1gradients_1/deep_component/dropout_1/mul_grad/Sum3gradients_1/deep_component/dropout_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_1/mul_grad/Mul_1Muldeep_component/ReluHgradients_1/deep_component/dropout_1/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:????????? 
?
3gradients_1/deep_component/dropout_1/mul_grad/Sum_1Sum3gradients_1/deep_component/dropout_1/mul_grad/Mul_1Egradients_1/deep_component/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients_1/deep_component/dropout_1/mul_grad/Reshape_1Reshape3gradients_1/deep_component/dropout_1/mul_grad/Sum_15gradients_1/deep_component/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
>gradients_1/deep_component/dropout_1/mul_grad/tuple/group_depsNoOp6^gradients_1/deep_component/dropout_1/mul_grad/Reshape8^gradients_1/deep_component/dropout_1/mul_grad/Reshape_1
?
Fgradients_1/deep_component/dropout_1/mul_grad/tuple/control_dependencyIdentity5gradients_1/deep_component/dropout_1/mul_grad/Reshape?^gradients_1/deep_component/dropout_1/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/deep_component/dropout_1/mul_grad/Reshape*'
_output_shapes
:????????? 
?
Hgradients_1/deep_component/dropout_1/mul_grad/tuple/control_dependency_1Identity7gradients_1/deep_component/dropout_1/mul_grad/Reshape_1?^gradients_1/deep_component/dropout_1/mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/deep_component/dropout_1/mul_grad/Reshape_1*
_output_shapes
: 
?
-gradients_1/deep_component/Relu_grad/ReluGradReluGradFgradients_1/deep_component/dropout_1/mul_grad/tuple/control_dependencydeep_component/Relu*
T0*'
_output_shapes
:????????? 
~
)gradients_1/deep_component/Add_grad/ShapeShapedeep_component/MatMul*
T0*
out_type0*
_output_shapes
:
|
+gradients_1/deep_component/Add_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
9gradients_1/deep_component/Add_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_1/deep_component/Add_grad/Shape+gradients_1/deep_component/Add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
'gradients_1/deep_component/Add_grad/SumSum-gradients_1/deep_component/Relu_grad/ReluGrad9gradients_1/deep_component/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
+gradients_1/deep_component/Add_grad/ReshapeReshape'gradients_1/deep_component/Add_grad/Sum)gradients_1/deep_component/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:????????? 
?
)gradients_1/deep_component/Add_grad/Sum_1Sum-gradients_1/deep_component/Relu_grad/ReluGrad;gradients_1/deep_component/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
-gradients_1/deep_component/Add_grad/Reshape_1Reshape)gradients_1/deep_component/Add_grad/Sum_1+gradients_1/deep_component/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
?
4gradients_1/deep_component/Add_grad/tuple/group_depsNoOp,^gradients_1/deep_component/Add_grad/Reshape.^gradients_1/deep_component/Add_grad/Reshape_1
?
<gradients_1/deep_component/Add_grad/tuple/control_dependencyIdentity+gradients_1/deep_component/Add_grad/Reshape5^gradients_1/deep_component/Add_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/deep_component/Add_grad/Reshape*'
_output_shapes
:????????? 
?
>gradients_1/deep_component/Add_grad/tuple/control_dependency_1Identity-gradients_1/deep_component/Add_grad/Reshape_15^gradients_1/deep_component/Add_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/deep_component/Add_grad/Reshape_1*
_output_shapes

: 
?
-gradients_1/deep_component/MatMul_grad/MatMulMatMul<gradients_1/deep_component/Add_grad/tuple/control_dependencyVariable/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
/gradients_1/deep_component/MatMul_grad/MatMul_1MatMuldeep_component/dropout/mul_1<gradients_1/deep_component/Add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	? 
?
7gradients_1/deep_component/MatMul_grad/tuple/group_depsNoOp.^gradients_1/deep_component/MatMul_grad/MatMul0^gradients_1/deep_component/MatMul_grad/MatMul_1
?
?gradients_1/deep_component/MatMul_grad/tuple/control_dependencyIdentity-gradients_1/deep_component/MatMul_grad/MatMul8^gradients_1/deep_component/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/deep_component/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
Agradients_1/deep_component/MatMul_grad/tuple/control_dependency_1Identity/gradients_1/deep_component/MatMul_grad/MatMul_18^gradients_1/deep_component/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/deep_component/MatMul_grad/MatMul_1*
_output_shapes
:	? 
?
3gradients_1/deep_component/dropout/mul_1_grad/ShapeShapedeep_component/dropout/mul*
T0*
out_type0*
_output_shapes
:
?
5gradients_1/deep_component/dropout/mul_1_grad/Shape_1Shapedeep_component/dropout/Cast*
T0*
out_type0*
_output_shapes
:
?
Cgradients_1/deep_component/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients_1/deep_component/dropout/mul_1_grad/Shape5gradients_1/deep_component/dropout/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
1gradients_1/deep_component/dropout/mul_1_grad/MulMul?gradients_1/deep_component/MatMul_grad/tuple/control_dependencydeep_component/dropout/Cast*
T0*(
_output_shapes
:??????????
?
1gradients_1/deep_component/dropout/mul_1_grad/SumSum1gradients_1/deep_component/dropout/mul_1_grad/MulCgradients_1/deep_component/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients_1/deep_component/dropout/mul_1_grad/ReshapeReshape1gradients_1/deep_component/dropout/mul_1_grad/Sum3gradients_1/deep_component/dropout/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
3gradients_1/deep_component/dropout/mul_1_grad/Mul_1Muldeep_component/dropout/mul?gradients_1/deep_component/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
3gradients_1/deep_component/dropout/mul_1_grad/Sum_1Sum3gradients_1/deep_component/dropout/mul_1_grad/Mul_1Egradients_1/deep_component/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
7gradients_1/deep_component/dropout/mul_1_grad/Reshape_1Reshape3gradients_1/deep_component/dropout/mul_1_grad/Sum_15gradients_1/deep_component/dropout/mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:??????????
?
>gradients_1/deep_component/dropout/mul_1_grad/tuple/group_depsNoOp6^gradients_1/deep_component/dropout/mul_1_grad/Reshape8^gradients_1/deep_component/dropout/mul_1_grad/Reshape_1
?
Fgradients_1/deep_component/dropout/mul_1_grad/tuple/control_dependencyIdentity5gradients_1/deep_component/dropout/mul_1_grad/Reshape?^gradients_1/deep_component/dropout/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/deep_component/dropout/mul_1_grad/Reshape*(
_output_shapes
:??????????
?
Hgradients_1/deep_component/dropout/mul_1_grad/tuple/control_dependency_1Identity7gradients_1/deep_component/dropout/mul_1_grad/Reshape_1?^gradients_1/deep_component/dropout/mul_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/deep_component/dropout/mul_1_grad/Reshape_1*(
_output_shapes
:??????????
?
1gradients_1/deep_component/dropout/mul_grad/ShapeShapedeep_component/Reshape*
T0*
out_type0*
_output_shapes
:
v
3gradients_1/deep_component/dropout/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
Agradients_1/deep_component/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients_1/deep_component/dropout/mul_grad/Shape3gradients_1/deep_component/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
/gradients_1/deep_component/dropout/mul_grad/MulMulFgradients_1/deep_component/dropout/mul_1_grad/tuple/control_dependencydeep_component/dropout/truediv*
T0*(
_output_shapes
:??????????
?
/gradients_1/deep_component/dropout/mul_grad/SumSum/gradients_1/deep_component/dropout/mul_grad/MulAgradients_1/deep_component/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
3gradients_1/deep_component/dropout/mul_grad/ReshapeReshape/gradients_1/deep_component/dropout/mul_grad/Sum1gradients_1/deep_component/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
1gradients_1/deep_component/dropout/mul_grad/Mul_1Muldeep_component/ReshapeFgradients_1/deep_component/dropout/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
1gradients_1/deep_component/dropout/mul_grad/Sum_1Sum1gradients_1/deep_component/dropout/mul_grad/Mul_1Cgradients_1/deep_component/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
5gradients_1/deep_component/dropout/mul_grad/Reshape_1Reshape1gradients_1/deep_component/dropout/mul_grad/Sum_13gradients_1/deep_component/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
<gradients_1/deep_component/dropout/mul_grad/tuple/group_depsNoOp4^gradients_1/deep_component/dropout/mul_grad/Reshape6^gradients_1/deep_component/dropout/mul_grad/Reshape_1
?
Dgradients_1/deep_component/dropout/mul_grad/tuple/control_dependencyIdentity3gradients_1/deep_component/dropout/mul_grad/Reshape=^gradients_1/deep_component/dropout/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/deep_component/dropout/mul_grad/Reshape*(
_output_shapes
:??????????
?
Fgradients_1/deep_component/dropout/mul_grad/tuple/control_dependency_1Identity5gradients_1/deep_component/dropout/mul_grad/Reshape_1=^gradients_1/deep_component/dropout/mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/deep_component/dropout/mul_grad/Reshape_1*
_output_shapes
: 
?
-gradients_1/deep_component/Reshape_grad/ShapeShapeembedding_lookup/Mul*
T0*
out_type0*
_output_shapes
:
?
/gradients_1/deep_component/Reshape_grad/ReshapeReshapeDgradients_1/deep_component/dropout/mul_grad/tuple/control_dependency-gradients_1/deep_component/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
gradients_1/AddN_1AddN&gradients_1/second_order/Sum_grad/Tilegradients_1/Square_grad/Mul_1/gradients_1/deep_component/Reshape_grad/Reshape*
T0*9
_class/
-+loc:@gradients_1/second_order/Sum_grad/Tile*
N*+
_output_shapes
:?????????
?
+gradients_1/embedding_lookup/Mul_grad/ShapeShape*embedding_lookup/embedding_lookup/Identity*
T0*
out_type0*
_output_shapes
:
?
-gradients_1/embedding_lookup/Mul_grad/Shape_1Shapeembedding_lookup/Reshape*
T0*
out_type0*
_output_shapes
:
?
;gradients_1/embedding_lookup/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients_1/embedding_lookup/Mul_grad/Shape-gradients_1/embedding_lookup/Mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
)gradients_1/embedding_lookup/Mul_grad/MulMulgradients_1/AddN_1embedding_lookup/Reshape*
T0*+
_output_shapes
:?????????
?
)gradients_1/embedding_lookup/Mul_grad/SumSum)gradients_1/embedding_lookup/Mul_grad/Mul;gradients_1/embedding_lookup/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
-gradients_1/embedding_lookup/Mul_grad/ReshapeReshape)gradients_1/embedding_lookup/Mul_grad/Sum+gradients_1/embedding_lookup/Mul_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :??????????????????
?
+gradients_1/embedding_lookup/Mul_grad/Mul_1Mul*embedding_lookup/embedding_lookup/Identitygradients_1/AddN_1*
T0*+
_output_shapes
:?????????
?
+gradients_1/embedding_lookup/Mul_grad/Sum_1Sum+gradients_1/embedding_lookup/Mul_grad/Mul_1=gradients_1/embedding_lookup/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
/gradients_1/embedding_lookup/Mul_grad/Reshape_1Reshape+gradients_1/embedding_lookup/Mul_grad/Sum_1-gradients_1/embedding_lookup/Mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:?????????
?
6gradients_1/embedding_lookup/Mul_grad/tuple/group_depsNoOp.^gradients_1/embedding_lookup/Mul_grad/Reshape0^gradients_1/embedding_lookup/Mul_grad/Reshape_1
?
>gradients_1/embedding_lookup/Mul_grad/tuple/control_dependencyIdentity-gradients_1/embedding_lookup/Mul_grad/Reshape7^gradients_1/embedding_lookup/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/embedding_lookup/Mul_grad/Reshape*4
_output_shapes"
 :??????????????????
?
@gradients_1/embedding_lookup/Mul_grad/tuple/control_dependency_1Identity/gradients_1/embedding_lookup/Mul_grad/Reshape_17^gradients_1/embedding_lookup/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_1/embedding_lookup/Mul_grad/Reshape_1*+
_output_shapes
:?????????
?
8gradients_1/embedding_lookup/embedding_lookup_grad/ShapeConst*%
_class
loc:@feature_embeddings*%
valueB	"?              *
dtype0	*
_output_shapes
:
?
7gradients_1/embedding_lookup/embedding_lookup_grad/CastCast8gradients_1/embedding_lookup/embedding_lookup_grad/Shape*

SrcT0	*%
_class
loc:@feature_embeddings*
Truncate( *

DstT0*
_output_shapes
:
|
7gradients_1/embedding_lookup/embedding_lookup_grad/SizeSize
feat_index*
T0*
out_type0*
_output_shapes
: 
?
Agradients_1/embedding_lookup/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
=gradients_1/embedding_lookup/embedding_lookup_grad/ExpandDims
ExpandDims7gradients_1/embedding_lookup/embedding_lookup_grad/SizeAgradients_1/embedding_lookup/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
?
Fgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Hgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
?
Hgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
@gradients_1/embedding_lookup/embedding_lookup_grad/strided_sliceStridedSlice7gradients_1/embedding_lookup/embedding_lookup_grad/CastFgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stackHgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stack_1Hgradients_1/embedding_lookup/embedding_lookup_grad/strided_slice/stack_2*
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
>gradients_1/embedding_lookup/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
9gradients_1/embedding_lookup/embedding_lookup_grad/concatConcatV2=gradients_1/embedding_lookup/embedding_lookup_grad/ExpandDims@gradients_1/embedding_lookup/embedding_lookup_grad/strided_slice>gradients_1/embedding_lookup/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
:gradients_1/embedding_lookup/embedding_lookup_grad/ReshapeReshape>gradients_1/embedding_lookup/Mul_grad/tuple/control_dependency9gradients_1/embedding_lookup/embedding_lookup_grad/concat*
T0*
Tshape0*'
_output_shapes
:?????????
?
<gradients_1/embedding_lookup/embedding_lookup_grad/Reshape_1Reshape
feat_index=gradients_1/embedding_lookup/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:?????????
Y
Adam_1/learning_rateConst*
valueB
 *o?:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
'Adam_1/update_feature_embeddings/UniqueUnique<gradients_1/embedding_lookup/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*%
_class
loc:@feature_embeddings*2
_output_shapes 
:?????????:?????????
?
&Adam_1/update_feature_embeddings/ShapeShape'Adam_1/update_feature_embeddings/Unique*
T0*%
_class
loc:@feature_embeddings*
out_type0*
_output_shapes
:
?
4Adam_1/update_feature_embeddings/strided_slice/stackConst*%
_class
loc:@feature_embeddings*
valueB: *
dtype0*
_output_shapes
:
?
6Adam_1/update_feature_embeddings/strided_slice/stack_1Const*%
_class
loc:@feature_embeddings*
valueB:*
dtype0*
_output_shapes
:
?
6Adam_1/update_feature_embeddings/strided_slice/stack_2Const*%
_class
loc:@feature_embeddings*
valueB:*
dtype0*
_output_shapes
:
?
.Adam_1/update_feature_embeddings/strided_sliceStridedSlice&Adam_1/update_feature_embeddings/Shape4Adam_1/update_feature_embeddings/strided_slice/stack6Adam_1/update_feature_embeddings/strided_slice/stack_16Adam_1/update_feature_embeddings/strided_slice/stack_2*
T0*
Index0*%
_class
loc:@feature_embeddings*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
3Adam_1/update_feature_embeddings/UnsortedSegmentSumUnsortedSegmentSum:gradients_1/embedding_lookup/embedding_lookup_grad/Reshape)Adam_1/update_feature_embeddings/Unique:1.Adam_1/update_feature_embeddings/strided_slice*
Tnumsegments0*
Tindices0*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
&Adam_1/update_feature_embeddings/sub/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
$Adam_1/update_feature_embeddings/subSub&Adam_1/update_feature_embeddings/sub/xbeta2_power/read*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
%Adam_1/update_feature_embeddings/SqrtSqrt$Adam_1/update_feature_embeddings/sub*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
$Adam_1/update_feature_embeddings/mulMulAdam_1/learning_rate%Adam_1/update_feature_embeddings/Sqrt*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
(Adam_1/update_feature_embeddings/sub_1/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
&Adam_1/update_feature_embeddings/sub_1Sub(Adam_1/update_feature_embeddings/sub_1/xbeta1_power/read*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
(Adam_1/update_feature_embeddings/truedivRealDiv$Adam_1/update_feature_embeddings/mul&Adam_1/update_feature_embeddings/sub_1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
(Adam_1/update_feature_embeddings/sub_2/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
&Adam_1/update_feature_embeddings/sub_2Sub(Adam_1/update_feature_embeddings/sub_2/xAdam_1/beta1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
&Adam_1/update_feature_embeddings/mul_1Mul3Adam_1/update_feature_embeddings/UnsortedSegmentSum&Adam_1/update_feature_embeddings/sub_2*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
&Adam_1/update_feature_embeddings/mul_2Mulfeature_embeddings/Adam/readAdam_1/beta1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
'Adam_1/update_feature_embeddings/AssignAssignfeature_embeddings/Adam&Adam_1/update_feature_embeddings/mul_2*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
+Adam_1/update_feature_embeddings/ScatterAdd
ScatterAddfeature_embeddings/Adam'Adam_1/update_feature_embeddings/Unique&Adam_1/update_feature_embeddings/mul_1(^Adam_1/update_feature_embeddings/Assign*
use_locking( *
Tindices0*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
&Adam_1/update_feature_embeddings/mul_3Mul3Adam_1/update_feature_embeddings/UnsortedSegmentSum3Adam_1/update_feature_embeddings/UnsortedSegmentSum*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
(Adam_1/update_feature_embeddings/sub_3/xConst*%
_class
loc:@feature_embeddings*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
&Adam_1/update_feature_embeddings/sub_3Sub(Adam_1/update_feature_embeddings/sub_3/xAdam_1/beta2*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
: 
?
&Adam_1/update_feature_embeddings/mul_4Mul&Adam_1/update_feature_embeddings/mul_3&Adam_1/update_feature_embeddings/sub_3*
T0*%
_class
loc:@feature_embeddings*'
_output_shapes
:?????????
?
&Adam_1/update_feature_embeddings/mul_5Mulfeature_embeddings/Adam_1/readAdam_1/beta2*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
)Adam_1/update_feature_embeddings/Assign_1Assignfeature_embeddings/Adam_1&Adam_1/update_feature_embeddings/mul_5*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
-Adam_1/update_feature_embeddings/ScatterAdd_1
ScatterAddfeature_embeddings/Adam_1'Adam_1/update_feature_embeddings/Unique&Adam_1/update_feature_embeddings/mul_4*^Adam_1/update_feature_embeddings/Assign_1*
use_locking( *
Tindices0*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
'Adam_1/update_feature_embeddings/Sqrt_1Sqrt-Adam_1/update_feature_embeddings/ScatterAdd_1*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
&Adam_1/update_feature_embeddings/mul_6Mul(Adam_1/update_feature_embeddings/truediv+Adam_1/update_feature_embeddings/ScatterAdd*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
$Adam_1/update_feature_embeddings/addAdd'Adam_1/update_feature_embeddings/Sqrt_1Adam_1/epsilon*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
*Adam_1/update_feature_embeddings/truediv_1RealDiv&Adam_1/update_feature_embeddings/mul_6$Adam_1/update_feature_embeddings/add*
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
*Adam_1/update_feature_embeddings/AssignSub	AssignSubfeature_embeddings*Adam_1/update_feature_embeddings/truediv_1*
use_locking( *
T0*%
_class
loc:@feature_embeddings*
_output_shapes
:	?
?
+Adam_1/update_feature_embeddings/group_depsNoOp+^Adam_1/update_feature_embeddings/AssignSub,^Adam_1/update_feature_embeddings/ScatterAdd.^Adam_1/update_feature_embeddings/ScatterAdd_1*%
_class
loc:@feature_embeddings
?
!Adam_1/update_feature_bias/UniqueUnique7gradients_1/first_order/embedding_lookup_grad/Reshape_1*
out_idx0*
T0*
_class
loc:@feature_bias*2
_output_shapes 
:?????????:?????????
?
 Adam_1/update_feature_bias/ShapeShape!Adam_1/update_feature_bias/Unique*
T0*
_class
loc:@feature_bias*
out_type0*
_output_shapes
:
?
.Adam_1/update_feature_bias/strided_slice/stackConst*
_class
loc:@feature_bias*
valueB: *
dtype0*
_output_shapes
:
?
0Adam_1/update_feature_bias/strided_slice/stack_1Const*
_class
loc:@feature_bias*
valueB:*
dtype0*
_output_shapes
:
?
0Adam_1/update_feature_bias/strided_slice/stack_2Const*
_class
loc:@feature_bias*
valueB:*
dtype0*
_output_shapes
:
?
(Adam_1/update_feature_bias/strided_sliceStridedSlice Adam_1/update_feature_bias/Shape.Adam_1/update_feature_bias/strided_slice/stack0Adam_1/update_feature_bias/strided_slice/stack_10Adam_1/update_feature_bias/strided_slice/stack_2*
T0*
Index0*
_class
loc:@feature_bias*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
-Adam_1/update_feature_bias/UnsortedSegmentSumUnsortedSegmentSum5gradients_1/first_order/embedding_lookup_grad/Reshape#Adam_1/update_feature_bias/Unique:1(Adam_1/update_feature_bias/strided_slice*
Tnumsegments0*
Tindices0*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
 Adam_1/update_feature_bias/sub/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Adam_1/update_feature_bias/subSub Adam_1/update_feature_bias/sub/xbeta2_power/read*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam_1/update_feature_bias/SqrtSqrtAdam_1/update_feature_bias/sub*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
Adam_1/update_feature_bias/mulMulAdam_1/learning_rateAdam_1/update_feature_bias/Sqrt*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
"Adam_1/update_feature_bias/sub_1/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 Adam_1/update_feature_bias/sub_1Sub"Adam_1/update_feature_bias/sub_1/xbeta1_power/read*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
"Adam_1/update_feature_bias/truedivRealDivAdam_1/update_feature_bias/mul Adam_1/update_feature_bias/sub_1*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
"Adam_1/update_feature_bias/sub_2/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 Adam_1/update_feature_bias/sub_2Sub"Adam_1/update_feature_bias/sub_2/xAdam_1/beta1*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
 Adam_1/update_feature_bias/mul_1Mul-Adam_1/update_feature_bias/UnsortedSegmentSum Adam_1/update_feature_bias/sub_2*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
 Adam_1/update_feature_bias/mul_2Mulfeature_bias/Adam/readAdam_1/beta1*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
!Adam_1/update_feature_bias/AssignAssignfeature_bias/Adam Adam_1/update_feature_bias/mul_2*
use_locking( *
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
%Adam_1/update_feature_bias/ScatterAdd
ScatterAddfeature_bias/Adam!Adam_1/update_feature_bias/Unique Adam_1/update_feature_bias/mul_1"^Adam_1/update_feature_bias/Assign*
use_locking( *
Tindices0*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
 Adam_1/update_feature_bias/mul_3Mul-Adam_1/update_feature_bias/UnsortedSegmentSum-Adam_1/update_feature_bias/UnsortedSegmentSum*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
"Adam_1/update_feature_bias/sub_3/xConst*
_class
loc:@feature_bias*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 Adam_1/update_feature_bias/sub_3Sub"Adam_1/update_feature_bias/sub_3/xAdam_1/beta2*
T0*
_class
loc:@feature_bias*
_output_shapes
: 
?
 Adam_1/update_feature_bias/mul_4Mul Adam_1/update_feature_bias/mul_3 Adam_1/update_feature_bias/sub_3*
T0*
_class
loc:@feature_bias*'
_output_shapes
:?????????
?
 Adam_1/update_feature_bias/mul_5Mulfeature_bias/Adam_1/readAdam_1/beta2*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
#Adam_1/update_feature_bias/Assign_1Assignfeature_bias/Adam_1 Adam_1/update_feature_bias/mul_5*
use_locking( *
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
'Adam_1/update_feature_bias/ScatterAdd_1
ScatterAddfeature_bias/Adam_1!Adam_1/update_feature_bias/Unique Adam_1/update_feature_bias/mul_4$^Adam_1/update_feature_bias/Assign_1*
use_locking( *
Tindices0*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
!Adam_1/update_feature_bias/Sqrt_1Sqrt'Adam_1/update_feature_bias/ScatterAdd_1*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
 Adam_1/update_feature_bias/mul_6Mul"Adam_1/update_feature_bias/truediv%Adam_1/update_feature_bias/ScatterAdd*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
Adam_1/update_feature_bias/addAdd!Adam_1/update_feature_bias/Sqrt_1Adam_1/epsilon*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
$Adam_1/update_feature_bias/truediv_1RealDiv Adam_1/update_feature_bias/mul_6Adam_1/update_feature_bias/add*
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
$Adam_1/update_feature_bias/AssignSub	AssignSubfeature_bias$Adam_1/update_feature_bias/truediv_1*
use_locking( *
T0*
_class
loc:@feature_bias*
_output_shapes
:	?
?
%Adam_1/update_feature_bias/group_depsNoOp%^Adam_1/update_feature_bias/AssignSub&^Adam_1/update_feature_bias/ScatterAdd(^Adam_1/update_feature_bias/ScatterAdd_1*
_class
loc:@feature_bias
?
 Adam_1/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/deep_component/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *
_output_shapes
:	? 
?
"Adam_1/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon>gradients_1/deep_component/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *
_output_shapes

: 
?
"Adam_1/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/deep_component/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *
_output_shapes

:  
?
"Adam_1/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/deep_component/Add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( *
_output_shapes

: 
?
"Adam_1/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon6gradients_1/output/Mul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( *
_output_shapes

:<
?
"Adam_1/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon6gradients_1/output/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *
_output_shapes
: 
?

Adam_1/mulMulbeta1_power/readAdam_1/beta1!^Adam_1/update_Variable/ApplyAdam#^Adam_1/update_Variable_1/ApplyAdam#^Adam_1/update_Variable_2/ApplyAdam#^Adam_1/update_Variable_3/ApplyAdam#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam&^Adam_1/update_feature_bias/group_deps,^Adam_1/update_feature_embeddings/group_deps*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam_1/AssignAssignbeta1_power
Adam_1/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
Adam_1/mul_1Mulbeta2_power/readAdam_1/beta2!^Adam_1/update_Variable/ApplyAdam#^Adam_1/update_Variable_1/ApplyAdam#^Adam_1/update_Variable_2/ApplyAdam#^Adam_1/update_Variable_3/ApplyAdam#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam&^Adam_1/update_feature_bias/group_deps,^Adam_1/update_feature_embeddings/group_deps*
T0*
_class
loc:@Variable*
_output_shapes
: 
?
Adam_1/Assign_1Assignbeta2_powerAdam_1/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
Adam_1/updateNoOp^Adam_1/Assign^Adam_1/Assign_1!^Adam_1/update_Variable/ApplyAdam#^Adam_1/update_Variable_1/ApplyAdam#^Adam_1/update_Variable_2/ApplyAdam#^Adam_1/update_Variable_3/ApplyAdam#^Adam_1/update_Variable_4/ApplyAdam#^Adam_1/update_Variable_5/ApplyAdam&^Adam_1/update_feature_bias/group_deps,^Adam_1/update_feature_embeddings/group_deps
~
Adam_1/valueConst^Adam_1/update*
_class
loc:@global_step*
value	B :*
dtype0*
_output_shapes
: 
?
Adam_1	AssignAddglobal_stepAdam_1/value*
use_locking( *
T0*
_class
loc:@global_step*
_output_shapes
: 
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
?
save/SaveV2/tensor_namesConst*?
value?B?BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1Bbeta1_powerBbeta2_powerBfeature_biasBfeature_bias/AdamBfeature_bias/Adam_1Bfeature_embeddingsBfeature_embeddings/AdamBfeature_embeddings/Adam_1Bglobal_step*
dtype0*
_output_shapes
:
?
save/SaveV2/shape_and_slicesConst*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_powerbeta2_powerfeature_biasfeature_bias/Adamfeature_bias/Adam_1feature_embeddingsfeature_embeddings/Adamfeature_embeddings/Adam_1global_step*)
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst*?
value?B?BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1Bbeta1_powerBbeta2_powerBfeature_biasBfeature_bias/AdamBfeature_bias/Adam_1Bfeature_embeddingsBfeature_embeddings/AdamBfeature_embeddings/Adam_1Bglobal_step*
dtype0*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*)
dtypes
2*?
_output_shapesn
l:::::::::::::::::::::::::::
?
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save/Assign_3Assign
Variable_1save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save/Assign_6Assign
Variable_2save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save/Assign_7AssignVariable_2/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save/Assign_9Assign
Variable_3save/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save/Assign_10AssignVariable_3/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save/Assign_12Assign
Variable_4save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save/Assign_13AssignVariable_4/Adamsave/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save/Assign_15Assign
Variable_5save/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save/Assign_16AssignVariable_5/Adamsave/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save/Assign_18Assignbeta1_powersave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
save/Assign_19Assignbeta2_powersave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
save/Assign_20Assignfeature_biassave/RestoreV2:20*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save/Assign_21Assignfeature_bias/Adamsave/RestoreV2:21*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save/Assign_22Assignfeature_bias/Adam_1save/RestoreV2:22*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save/Assign_23Assignfeature_embeddingssave/RestoreV2:23*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save/Assign_24Assignfeature_embeddings/Adamsave/RestoreV2:24*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save/Assign_25Assignfeature_embeddings/Adam_1save/RestoreV2:25*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save/Assign_26Assignglobal_stepsave/RestoreV2:26*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
?
initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^beta1_power/Assign^beta2_power/Assign^feature_bias/Adam/Assign^feature_bias/Adam_1/Assign^feature_bias/Assign^feature_embeddings/Adam/Assign!^feature_embeddings/Adam_1/Assign^feature_embeddings/Assign^global_step/Assign

init_1NoOp
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_ef31c8a5eae24d6a8e2f984738a092de/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst*?
value?B?BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1Bbeta1_powerBbeta2_powerBfeature_biasBfeature_bias/AdamBfeature_bias/Adam_1Bfeature_embeddingsBfeature_embeddings/AdamBfeature_embeddings/Adam_1Bglobal_step*
dtype0*
_output_shapes
:
?
save_1/SaveV2/shape_and_slicesConst*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_powerbeta2_powerfeature_biasfeature_bias/Adamfeature_bias/Adam_1feature_embeddingsfeature_embeddings/Adamfeature_embeddings/Adam_1global_step*)
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst*?
value?B?BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1Bbeta1_powerBbeta2_powerBfeature_biasBfeature_bias/AdamBfeature_bias/Adam_1Bfeature_embeddingsBfeature_embeddings/AdamBfeature_embeddings/Adam_1Bglobal_step*
dtype0*
_output_shapes
:
?
!save_1/RestoreV2/shape_and_slicesConst*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*)
dtypes
2*?
_output_shapesn
l:::::::::::::::::::::::::::
?
save_1/AssignAssignVariablesave_1/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save_1/Assign_1AssignVariable/Adamsave_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save_1/Assign_2AssignVariable/Adam_1save_1/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	? 
?
save_1/Assign_3Assign
Variable_1save_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_4AssignVariable_1/Adamsave_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_5AssignVariable_1/Adam_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_6Assign
Variable_2save_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save_1/Assign_7AssignVariable_2/Adamsave_1/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save_1/Assign_8AssignVariable_2/Adam_1save_1/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes

:  
?
save_1/Assign_9Assign
Variable_3save_1/RestoreV2:9*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_10AssignVariable_3/Adamsave_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_11AssignVariable_3/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes

: 
?
save_1/Assign_12Assign
Variable_4save_1/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save_1/Assign_13AssignVariable_4/Adamsave_1/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save_1/Assign_14AssignVariable_4/Adam_1save_1/RestoreV2:14*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:<
?
save_1/Assign_15Assign
Variable_5save_1/RestoreV2:15*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_16AssignVariable_5/Adamsave_1/RestoreV2:16*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_17AssignVariable_5/Adam_1save_1/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_18Assignbeta1_powersave_1/RestoreV2:18*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_19Assignbeta2_powersave_1/RestoreV2:19*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_20Assignfeature_biassave_1/RestoreV2:20*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_21Assignfeature_bias/Adamsave_1/RestoreV2:21*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_22Assignfeature_bias/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*
_class
loc:@feature_bias*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_23Assignfeature_embeddingssave_1/RestoreV2:23*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_24Assignfeature_embeddings/Adamsave_1/RestoreV2:24*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_25Assignfeature_embeddings/Adam_1save_1/RestoreV2:25*
use_locking(*
T0*%
_class
loc:@feature_embeddings*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_26Assignglobal_stepsave_1/RestoreV2:26*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard "&B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
	variables??
_
feature_embeddings:0feature_embeddings/Assignfeature_embeddings/read:02random_normal:08
N
feature_bias:0feature_bias/Assignfeature_bias/read:02random_uniform:08
J

Variable:0Variable/AssignVariable/read:02Variable/initial_value:08
R
Variable_1:0Variable_1/AssignVariable_1/read:02Variable_1/initial_value:08
R
Variable_2:0Variable_2/AssignVariable_2/read:02Variable_2/initial_value:08
R
Variable_3:0Variable_3/AssignVariable_3/read:02Variable_3/initial_value:08
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
?
Variable_5:0Variable_5/AssignVariable_5/read:02Const:08
T
global_step:0global_step/Assignglobal_step/read:02global_step/initial_value:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
feature_embeddings/Adam:0feature_embeddings/Adam/Assignfeature_embeddings/Adam/read:02+feature_embeddings/Adam/Initializer/zeros:0
?
feature_embeddings/Adam_1:0 feature_embeddings/Adam_1/Assign feature_embeddings/Adam_1/read:02-feature_embeddings/Adam_1/Initializer/zeros:0
p
feature_bias/Adam:0feature_bias/Adam/Assignfeature_bias/Adam/read:02%feature_bias/Adam/Initializer/zeros:0
x
feature_bias/Adam_1:0feature_bias/Adam_1/Assignfeature_bias/Adam_1/read:02'feature_bias/Adam_1/Initializer/zeros:0
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
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0"?
trainable_variables??
_
feature_embeddings:0feature_embeddings/Assignfeature_embeddings/read:02random_normal:08
N
feature_bias:0feature_bias/Assignfeature_bias/read:02random_uniform:08
J

Variable:0Variable/AssignVariable/read:02Variable/initial_value:08
R
Variable_1:0Variable_1/AssignVariable_1/read:02Variable_1/initial_value:08
R
Variable_2:0Variable_2/AssignVariable_2/read:02Variable_2/initial_value:08
R
Variable_3:0Variable_3/AssignVariable_3/read:02Variable_3/initial_value:08
R
Variable_4:0Variable_4/AssignVariable_4/read:02Variable_4/initial_value:08
?
Variable_5:0Variable_5/AssignVariable_5/read:02Const:08"#
losses

loss/log_loss/value:0"
train_op

Adam
Adam_1*?
serving_default?
2
X1,
feat_index:0??????????????????
2
Xv,
feat_value:0??????????????????
7
dropout_keep_fm$
dropout_keep_fm:0?????????
;
dropout_keep_deep&
dropout_keep_deep:0?????????+
out$
sigmoid_out:0?????????tensorflow/serving/predict