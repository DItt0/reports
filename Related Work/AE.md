**Autoencoder in autoencoder networks**  
该论文介绍了一种新的无监督多视图表示学习算法，称为autoencoder in autoencoder networks (AE2-Nets)。其主要方法是通过双向编码策略，有效地将高维异构数据中的信息编码为紧凑且富有信息的表示。
具体而言，AE2-Nets采用两种编码方向：内部的AE网络（inner-AE-networks）负责提取每个视图特定的内在信息（正向编码）；外部的AE网络（outer-AE-networks）则将来自不同视图的内在信息整合到一个潜在表示中（反向编码）。该嵌套架构进一步通过层次变分自编码器（hierarchical variational autoencoder）提供了概率解释和扩展。
这种正向-反向编码策略灵活地解决了每个视图中的高维（噪声）特征问题，并在统一框架下编码了多视图间的互补信息。
总结：该方法的关键在于使用了自编码器嵌套结构，通过正向和反向的编码策略来处理高维和噪声特征，同时整合多个视图的数据。  

**GAMC: an unsupervised method for fake news detection using graph autoencoder with masking**  
该论文提出了一种名为GAMC的无监督假新闻检测技术，利用**图自编码器（Graph Autoencoder）与掩蔽对比学习（Masking and Contrastive learning）**相结合，旨在解决传统方法中对标注数据的依赖问题。其核心思想是通过结合新闻传播的内容和上下文信息作为自监督信号，从而减少对大规模标注数据的需求。
具体而言，GAMC方法首先对原始新闻传播图进行数据增强，生成增强后的图结构。接着，使用图编码器（graph encoder）对这些增强后的图进行编码，并通过图解码器（graph decoder）进行重构。该过程通过设计一个复合损失函数来进行优化。这个损失函数包括重构误差和对比损失，主要有两个作用：一方面，最小化重构误差，以确保模型能够有效捕获潜在特征；另一方面，对比损失用于对齐来自同一源的增强图的表示，增强模型的区分能力。
总结：该方法结合了图自编码器和对比学习，通过无监督的方式进行假新闻检测，减少了对标注数据的依赖，并通过图的增强和重构捕获新闻传播的潜在信息。  

**ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation** 
该论文提出了一种名为ContrastVAE的模型，将**对比学习（Contrastive Learning）与变分自编码器（Variational AutoEncoder, VAE）结合，旨在解决现有序列推荐方法中的一些问题，特别是用户与物品交互的稀疏性、不确定性以及长尾问题。
ContrastVAE的核心创新是引入了一个新的训练目标——ContrastELBO。这一目标将传统的单视角ELBO（变分自编码器的训练目标）扩展为双视角问题，并从理论上构建了VAE与对比学习之间的联系。为了使对比学习得以实施，作者提出了模型增强（model augmentation）和变分增强（variational augmentation）**两种简单有效的增强策略，生成序列的第二视角，从而引入对比学习。
实验结果表明，ContrastVAE及其增强方法在四个基准数据集上的表现都显著优于其他现有方法。  

**Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms**  
该论文提出了一种新的方法——语义金字塔自编码器（SPAE），旨在使**冻结的语言大模型（LLM）**能够执行涉及非语言模态（如图像或视频）的理解和生成任务。SPAE通过将原始像素转换为从LLM词汇表中提取的可解释的词汇符号（或单词）来工作。生成的符号不仅捕获了语义信息，还包含了进行视觉重建所需的细节，从而有效地将视觉内容翻译成LLM可理解的语言，进而赋予LLM执行各种多模态任务的能力。  

**Graph Masked Autoencoder for Sequential Recommendation**  
这篇论文的摘要中提到了AutoEncoder。具体来说，提出的MAERec方法使用了图掩码自编码器（Graph Masked AutoEncoder, MAE），并通过自监督增强的方式进行数据重构，来建模物品之间的长程依赖关系。该方法利用自适应的数据重构框架进行增强，而不是依赖手工设计的对比视图生成策略。  

**Dual Low-Rank Graph Autoencoder for Semantic and Topological Networks.**  
这篇论文的摘要中提到了AutoEncoder，具体是提出了Dual Low-Rank Graph AutoEncoder (DLR-GAE)，一种图神经网络模型，结合了语义图和拓扑图的信息。DLR-GAE通过探索这两种图之间的低秩信息，利用自编码器重构邻接矩阵。与传统方法不同，DLR-GAE没有共享权重，而是专门为两种图的关系进行建模，并通过设计替代物和投影来限制学习到的因子矩阵。  

**GiGaMAE: Generalizable Graph Masked Autoencoder via Collaborative Latent Space Reconstruction**  
这篇论文提出了一种新的图数据自监督学习框架——GiGaMAE，旨在解决当前掩蔽自编码器（Masked Autoencoder，MAE）在图数据上的泛化能力不足问题。与现有的掩蔽自编码器不同，GiGaMAE不是通过显式地重建图的原始组成部分（例如特征或边），而是通过协同重建信息丰富的集成潜在嵌入。该方法将包含图拓扑和属性信息的嵌入作为重建目标，从而捕获更为广泛和全面的知识。此外，GiGaMAE引入了基于互信息的重建损失函数，能够有效地重建多个目标。这一学习目标使得模型能够区分从单一目标学习到的独特知识和从多个目标中共享的通用知识。  













