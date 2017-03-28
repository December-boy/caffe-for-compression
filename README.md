# 说明
摘自http://blog.csdn.net/zhouyusong_bupt/article/details/51728910

    传统的CNN网络训练完之后，全连接层的权值矩阵动辄就几十万、几百万个参数值，可见CNN模型的庞大，但是仔细观察CNN的权值矩阵就会发现，里面有很多的参数的绝对值都很小，比如在-0.001到0.001之间，也就是说这些连接对CNN的训练或者测试结果作用很小，因此我们就可以尝试将这些小值参数去掉，既可以减小模型的规模又可以减少计算量，最重要的前提是要保证CNN的有效性，也即正确率。

## 主要思路

修改blob的结构，将原来的矩阵改写成稀疏矩阵的存储方式
采用新的方法来计算稀疏矩阵和向量的相乘

## 具体实现
### blob的修改

在这里需要对blob.hpp和blob.cpp进行修改：

#### blob.hpp的修改（include/caffe/blob.hpp）
在原来的blob中，存有data_、diff_、shape_data_、shape_、count_、capacity_这6个属性。因为我们要将原来的矩阵（后文为了区分称为密集矩阵）存储为稀疏矩阵，所以要添加新的属性来存储稀疏后的矩阵参数。稀疏矩阵的存储方式可以参考这里，在这里我们添加了3个向量csrval_、csrrowptr_、csrcolind_，这三个变量分别存储着所有非零元素值、非零元素行指针、非零元素列索引。除了这三个新的变量外，还需要添加三个变量nnz_、sparse_、mask_，nnz_用来存储非零元素的个数，sparse_用来表征data是否需要进行稀疏存储，第三个变量mask_需要重点说一下。在我们剪枝的过程中会把data中的一些元素置为零，大量的元素值为零之后势必会影响网络的准确性，所以需要重新训练，将剩下的非零权值进行一次再训练，为了保证在再训练过程中非零元素不会被反馈过程更改掉，我们需要加一个mask_,用来标示该元素是否需要进行梯度更新，该mask_在最初的初始化时应该全为1，在剪枝阶段进行更新。 
除了给blob添加新的属性之外，还需要给新加入的属性添加相应的set和get方法，添加方法时参考blob中data和diff的方法(由于源码太长，在此就不粘贴了，具体查看源码)。

#### blob.cpp的修改（src/caffe/blob.cpp）

首先将新添加的变量的get和set方法实现，这部分比较简单，基本上都是复制粘贴修改变量名。除此之外还有三个比较重要的函数：Update(),FromProto()和ToProto().

##### Update()

该函数主要用来在每次后向反馈之后对blob中的data参数进行更新，因为我们添加了mask_矩阵，所以需要在正常反馈之后将更新值屏蔽掉，于是我们在更新之后将data_和mask_的对应位相乘，屏蔽掉更新，在这里我们调用了caffe中的caffe_gpu_mul()方法,代码如下：
```c
if(sparse_&&FLAGS_step!="three")
    caffe_gpu_mul<Dtype>(count_,
        static_cast<const Dtype*>(mask_->gpu_data()),
        static_cast<const Dtype*>(data_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
```
上面代码最上方有一个if判断，sparse_用来判断当前blob是否是需要进行稀疏压缩的blob,FLAGS_step用来表征当前是第几阶段，如果是第三阶段，则不进行该过程。

##### FLAGS_Step

在介绍ToProto()和FromProto()之前，先介绍一下FLAGS_step。如果从整体上去观察我们的剪枝过程，可以将其分成三步： 
1. 常规训练CNN网络，并保存训练后的模型，然后将小值参数置为零 
2. 对置零后的网路进行再训练，保存最终的caffemodel 
3. 读入caffemodel进行测试 
我们将步骤一中最先保存的caffemodel记为origin，小值置为零的caffemodel记为fixed，将步骤二中再次训练好的caffemodel记为retune。这三个不同的caffemodel除了名字之外，还有很多不同，下面通过表格列举一下。

model  |data	|diff	|mask	|csr	|是否为稀疏矩阵
---|
origin	|保存	|不保存	|保存	|保存	|否
fixed	|保存	|不保存	|保存	|保存	|是
retune	|不保存	|不保存	|不保存	|保存	|是

注：上图中的’保存’代表：该caffemodel中保存了该项参数值，’csr’代表：csrval、csrrowptr和csrcolind这三个向量的总称。 
因为在剪枝的不同阶段生成的caffemodel是不同的，所以在将训练好的网络保存下来和读入时需要根据不同阶段区别对待。为了区别不同阶段，我们引入了FLAGS_step这个全局变量。该变量可以通过命令行读入，关于FLAGS_name形式的全局变量，可以参考这篇博文。

##### ToProto()

该函数定义了如何将网络训练的权值参数保存进caffemodel中，比如是否将diff_保存进caffemodel中。 
在该函数中最主要的修改是实现了对密集矩阵的稀疏处理，生成csrval_、csrrowptr_、csrcolind_和nnz_,将稀疏矩阵进行保存（关于如何生成的csr相关向量，我们单独放在下面一节说）。在将参数矩阵保存进caffemodel时，主要通过sparse_和FLAGS_step这两个变量进行控制。只有sparse_为true时，我们才会对当前blob进行稀疏化处理，否则只进行常规处理。当需要对该blob进行稀疏化处理时，只有FLAGS_step等于one的时候，才会保存data和mask，否则不保存这两个参数矩阵。

#### 稀疏矩阵的存储

稀疏矩阵的存储可以说是CNN剪枝的重点，在实现中，我们调用了CUDA的cuSPARSE库，该库主要是为了优化稀疏矩阵的运算，提供了很多方便易用的接口，在这里我们用到了它的cusparseSnnz(),cusparseSdense2csc(),cusparseScsrmv()这几个函数接口，cusparseSnnz()主要是用来求出矩阵的非零元素个数，cusparseSdense2csc主要是生成矩阵的csrval、csrrowptr和csrcolind这几个特征，cusparseScsrmv()主要是计算稀疏矩阵和向量相乘的，这个才是我们的最终目的，在这个函数中我们需要传入上面生成的那几个csr向量。在此有几个坑，我简单说一下。首先，在caffe中矩阵是行主序的，但是在cuda中矩阵式列主序的，行主序就是把矩阵一行一行的存入内存，列主序是把矩阵一列一列的存入内存，这也是为什么我用的是cusparseScsc()而不是cusparseScsr()；第二点是要注意CPU和GPU之间的数据交换，需要用相应的函数cudaMemcpy()去复制一份，否则会报错；第三点是cuSPARSE库的异步性，要想同步执行各个函数，需要明确指定，可以用cudaDeviceSynchronize();来完成。

##### FromProto()

与ToProto()相反，该函数主要是将权值矩阵从caffemodel中读出来，根据FLAGS_step和sparse_的不同，有选择的读出csr、data、mask等。在这个地方需要注意的是，因为blob的reshape()中没有对csr进行初始化，所以在进行读出csr时，需要先给csr申请空间，然后再读出。

### proto的修改

#### caffe.proto的修改（src/caffe/proto/caffe.proto）
caffe.proto主要是用来定义数据存储结构的，比如我们ToProto()时，caffe。proto中要有和caffe中blob相对应的存储结构，各个属性名也最好能对上，方便记忆。
```c
message BlobProto {
  optional BlobShape shape = 7;
  repeated float data = 5 [packed = true];
  repeated float diff = 6 [packed = true];
  repeated double double_data = 8 [packed = true];
  repeated double double_diff = 9 [packed = true];

  // 4D dimensions -- deprecated.  Use "shape" instead.
  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
  optional bool  sparse = 10 [default = false];

  repeated float mask = 11 [packed = true];
  repeated double double_mask =12 [packed =true];

  optional int32 nnz = 13 [default = 0];
  repeated float csrval=14 [packed =true];
  repeated int32 csrrowptr =15 [packed =true];
  repeated int32 csrcolind =16 [packed =true];

  repeated double double_csrval=17 [packed =true];
}
```
在上面的代码中，message是关键字，后面的BlobProto是类名，optional,repeated和required是限定符，每个变量后面的数字不能重复，一般是依次向后排。在这里，我们新添加了mask,double_mask，nnz,csrval,csrrowptr,csrcolind和double_csrval这几个变量，这几个变量都和blob中新添加的变量一一对应。 
除了BlobProto以外，还给FillerParameter添加了一个新的mvalue，
```c
optional float mvalue= 10 [default = 1];
```
### filler的修改

#### filler.hpp的修改（include/caffe/filler.hpp）
在Caffe框架下，网络的初始化有两种方式，一种是调用filler，按照模型中定义的初始化方式进行初始化，第二种是从已有的caffemodel或者snapshot中读取相应参数矩阵进行初始化。 
在利用第一种方式初始化时，我们需要对新加的mask进行赋值。在filler.hpp中，caffe先定义了一个Filler父类，然后定义了一些Filler的子类，比如：ConstantFiller、GaussianFiller、XavierFiller等。我们为了实现对mask的初始化，在Filler的父类中定义了一个新的方法，在方法中实现了mask的初始化。
```c
void FillMask(Blob<Dtype>* blob){
    if(!blob->sparse())return;
    Dtype* mask =blob->mutable_cpu_mask();
    const int count = blob->count();
    const Dtype mvalue = this->filler_param_.mvalue();
    CHECK(count);
    for(int i=0;i<count;i++){
        mask[i]=mvalue;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
```
然后在子类的Fill方法中，调用FillMask()方法完成mask的初始化。在这里需要注意一下FillMask()函数内部第一句的判断，意思是：只有当前blob的sparse属性为true，我们才会进行mask填充，否则返回。

### common的修改

这里的common指的是common.hpp和common.cpp这两个文件。因为在调用cuSPARSE库中的函数时，都会用到两个变量，我们在回顾一下cuSPARSE的API。
```c
cusparseStatus_t 
cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *x, const float *beta, float *y)
```
我们忽略int型和float型的参数，除此之外还剩下三个参数，分别为handle、transA、descrA，handle是调用cuSPARSE函数的句柄，每个函数都要传入的，而且handle的定义和赋值都有专门的构造函数，比较耗时；transA是一个枚举数，代表矩阵是否需要旋转，因为是枚举值，定义该变量的时间可以忽略；descrA是一个结构体，声明和定义也都需要专门的函数，同时比较耗费时间。虽然handle和descrA都比较耗时，但是在多次调用时，每次调用的handle和descrA的值都是一样的，所以我们可以考虑每次只定义一次handle和descrA，存为全局变量，每次调用时就可以节省很多时间。接下来的common修改就是要实现这个目的。

#### common.hpp的修改 (include/caffe/common.hpp)
common中定义了caffe类，可以全局调用，我们在caffe类里新定义了两个变量，如下，并且定义了这两个变量的get函数。
```c
cusparseMatDescr_t cusparse_descr_;
cusparseHandle_t cusparse_handle_;
```
```c
inline static cusparseHandle_t cusparse_handle(){ return Get().cusparse_handle_;}
inline static cusparseMatDescr_t cusparse_descr(){ return Get().cusparse_descr_;}
```

#### common.cpp的修改 (src/caffe/common.cpp)
common.cpp中主要实现了caffe类的定义，首先在caffe的初始化函数中添加两个新定义的变量的初始化以及在析构函数中的析构。
```c
if(cusparseCreate(&cusparse_handle_)!=CUSPARSE_STATUS_SUCCESS){
    LOG(ERROR) << "cannot create Cusparse handle,Cusparse won't be available.";
} 
if(cusparseCreateMatDescr(&cusparse_descr_)!=CUSPARSE_STATUS_SUCCESS){
    LOG(ERROR) << "cannot create Cusparse descr,descr won't be available.";
}else {
    cusparseSetMatType(cusparse_descr_,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparse_descr_,CUSPARSE_INDEX_BASE_ZERO);
    LOG(INFO)<<"init descr";
}
```
```c
if (cusparse_descr_) CUSPARSE_CHECK(cusparseDestroyMatDescr(cusparse_descr_));
if (cusparse_handle_) CUSPARSE_CHECK(cusparseDestroy(cusparse_handle_));
```
除此之外还要在SetDevice()中添加cusparse_descr_和cusparse_handle_的销毁和创建。
```c
if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
if (Get().cusparse_descr_)CUSPARSE_CHECK(cusparseDestroyMatDescr(Get().cusparse_descr_));
CUSPARSE_CHECK(cusparseCreate(&Get().cusparse_handle_));
CUSPARSE_CHECK(cusparseCreateMatDescr(&Get().cusparse_descr_));
```

### caffe的修改

#### caffe.cpp的修改 （tools/caffe.cpp）
因为我们的网络剪枝一共有三个阶段，每个阶段会有不同的FromProto()和ToProto()，所以需要在运行caffe的时候需要指定当前是哪个阶段，在前面我们说到用FLAGS_step这个变量表征，这个变量的具体定义就是在caffe.cpp中完成的。step一共可以取三个值，分别为one、two、three，分别代表了上文中说到的三个阶段，关于DEFINE_string()这个API怎么用，可以参照这里。
```c
DEFINE_string(step,"one",
        "optional;choose the type of proto:"
        "one,two or three");
```

#### caffe.hpp的修改 （include/caffe/caffe.hpp）
在caffe.cpp中，我们定义了FLAGS_step这个变量，但是想要在其他文件中使用，还需要声明一下，因为主要是blob.cpp中使用，我们可以在caffe.hpp中声明一下，然后在blob.cpp中将caffe.hpp引入就行了。FLAGS_step的声明方式也要遵循flags的标准。
```c
DECLARE_string(step);
```

### math_functions的修改
math_functions中主要定义了很多常用的计算函数，比如矩阵相乘、相加、相减、线性变化等，当然还包括矩阵和向量相乘的函数——caffe_gpu_gemv(),但是该函数传入的是密集矩阵和向量，我们需要在这里新定义一个函数，用来实现稀疏矩阵和向量的相乘——caffe_gpu_csrmv()。

#### math_functions.hpp的修改 （include/caffe/util/math_functions.hpp）
在math_functions.hpp中我们声明一个新的函数，函数名为caffe_gpu_csrmv(),相应的参数参照caffe_gpu_gemv()，具体如下：
```c
template <typename Dtype>
void caffe_gpu_csrmv(const CBLAS_TRANSPOSE TransA, const int M, const int N,const Dtype alpha, const Dtype* csrval,const int* csrrowptr,const int* csrcolind,const int nnz, const Dtype* x, const Dtype beta,Dtype* y);
```

#### math_functions.cu的修改 （src/caffe/util/math_functions.cu）
这里稍微注意一下是math_functions.cu而不是math_functions.cpp，在这个文件中，我们主要添加了csrmv函数的实现部分。
```c
template <>
void caffe_gpu_csrmv<float>(const CBLAS_TRANSPOSE TransA, const int M,
        const int N,const float alpha,const float* csrval,const int* csrrowptr,const int* csrcolind,const int nnz,const float* x,const float beta,float* y){
    cusparseOperation_t cuTransA=(TransA == CblasNoTrans)?CUSPARSE_OPERATION_NON_TRANSPOSE:CUSPARSE_OPERATION_TRANSPOSE;
    CUSPARSE_CHECK(cusparseScsrmv(Caffe::cusparse_handle(),cuTransA,M,N,nnz,&alpha,Caffe::cusparse_descr(),csrval,csrrowptr,csrcolind,x,&beta,y));
}
```
以上是关于float类型的csrmv函数的实现，double型的修改一下类型就OK了。

### inner_product_layer的修改

首先解释一下为什么单独拿出这个层进行修改，因为整个CNN网络中参数最多的地方就是这里的全连接层，所以我们进行的剪枝主要是在全连接层进行的，上面提到的添加mask或者csr向量也都是针对全连接层的。

#### inner_product_layer.cpp的修改 （src/caffe/layers/inner_product_layer.cpp）
在上文中，我们说到要给blob添加mask，但是并不是给所有的blob添加mask，在这里我们主要给全连接层进行了剪枝，所以在这一层的LayerSetup()中单独给blob_[0]添加了mask，并将其sparse属性置为真。主要修改如下：

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    this->blobs_[0]->setSparse();
    this->blobs_[0]->Addmask(weight_shape); 

#### inner_product_layer.cu的修改 （src/caffe/layers/inner_product_layer.cu）
因为全连接层的参数矩阵进行了稀疏，所以在进行前向传播计算矩阵和向量相乘时，应该调用新定义的caffe_gpu_csrmv()函数，具体修改如下：

```c
if(M_ == 1) {
  LOG(INFO)<<"here is csrmv";
  const Dtype* csrval = this->blobs_[0]->gpu_csrval();
  const int* csrrowptr = this->blobs_[0]->gpu_csrrowptr();
  const int* csrcolind = this->blobs_[0]->gpu_csrcolind();
  const int nnz=this->blobs_[0]->nnz();
  caffe_gpu_csrmv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,csrval,csrrowptr,csrcolind,nnz, bottom_data, (Dtype)0., top_data);
if(bias_term_)
     caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],this->blobs_[1]->gpu_data(), top_data);
}
```
### _caffe.cpp的修改

_caffe.cpp这个文件主要是定义了Python调用caffe各个功能的接口，因为我们在blob中新添加了一些属性和方法，所以相应的需要在这个文件里提供一个映射，让我们使用Python接口时可以找到新定义的属性和方法。

_caffe.cpp的修改 （python/caffe/_caffe.cpp）

这个文件的原理我也不是很清楚，基本上是照葫芦画瓢就可以了，在Blob这个代码块里，添加上mask、csrval、nnz等属性即可。
```c
bp::class_<Blob<Dtype>, shared_ptr<Blob<Dtype> >, boost::noncopyable>(
    "Blob", bp::no_init)
    .add_property("shape",
        bp::make_function(
            static_cast<const vector<int>& (Blob<Dtype>::*)() const>(
                &Blob<Dtype>::shape),
            bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("num",      &Blob<Dtype>::num)
    .add_property("channels", &Blob<Dtype>::channels)
    .add_property("height",   &Blob<Dtype>::height)
    .add_property("width",    &Blob<Dtype>::width)
    .add_property("count",    static_cast<int (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::count))
    .add_property("nnz",      static_cast<int (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::nnz))
    .add_property("sparse",   static_cast<bool (Blob<Dtype>::*)() const>(
        &Blob<Dtype>::sparse))
    .def("reshape",           bp::raw_function(&Blob_Reshape))
    .add_property("data",     bp::make_function(&Blob<Dtype>::mutable_cpu_data,
          NdarrayCallPolicies()))
    .add_property("diff",     bp::make_function(&Blob<Dtype>::mutable_cpu_diff,
          NdarrayCallPolicies()))
    .add_property("mask",     bp::make_function(&Blob<Dtype>::mutable_cpu_mask,
          NdarrayCallPolicies()))
    .add_property("csrval",   bp::make_function(&Blob<Dtype>::mutable_cpu_csrval,
          NdarrayCallPolicies()));

 bp::register_ptr_to_python<shared_ptr<Blob<Dtype> > >();
```

### 其他修改

除了上文中提到的比较大的修改，还有一些小的地方也得修改。 
Makefile文件的修改：要添加对cuSparse的编译，详情见代码

#### FLAGS_step的声明

在Caffe中原来定义的工具类中，还必须把新添加的FLAGS_step变量的声明加入到工具类中，相关文件名及其路径如下：
```c
tools/upgrade_net_proto_binary.cpp
tools/upgrade_net_proto_text.cpp
tools/upgrade_solver_proto_text.cpp
tools/extract_features.cpp
tools/compute_image_mean.cpp
examples/cifar10/convert_cifar_data.cpp
examples/cpp_classification/classification.cpp
examples/mnist/convert_mnist_data.cpp
examples/siamese/convert_mnist_siamese_data.cpp
```
### 关于CUDA版本

因为利用了cuSPARSE库，所以需要用CUDA v7.5来编译Caffe。

## 实验结果

关于实验结果，我们主要比较了两个指标：存储空间和运行速度。

### 存储空间

|model	|storage（before）	|storage(after)	|rate|
|------|----|------|------|
|LeNet	|1.7M	|340K	|5:1|
|类AlexNet|	99.6M	|16M	|6:1|
因为我们只对全连接层进行了剪枝，所以后面的类AlexNet的压缩比会更大一些，如果我们模型更加复杂，压缩比会更大一些，除此之外，对卷积层也进行剪枝的话，也会进一步提高压缩比。

### 运行速度

在LeNet上，速度提升不明显，可能是因为cuSPARSE库本身的开销比较大，在较小的网络上效果不明显。 
在类AlexNet上，剪枝后版本的速度是原来速度的2-4倍，同样的，如果网络更大一些，效果会更好。

## 总结

通过将CNN进行剪枝，的确可以在保证准确率的前途下，实现模型存储空间的压缩和运行速度的提升。




____________________________________________________________________________________________________________________
# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
