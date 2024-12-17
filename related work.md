**Harnessing Network Effect for Fake News Mitigation: Selecting Debunkers via Self-Imitation Learning**  
这篇文章研究如何通过在社交网络中部署辟谣者传播真实新闻，来减少假新闻的影响。作者将问题建模为强化学习问题，每一步选择一个用户来传播真实新闻。核心挑战是仅能观测到整体的缓解效果，而无法直接衡量单个辟谣者的贡献。现有的自模仿学习（SIL）方法在处理这种情景奖励方面有潜力，但采样效率低，不适用于假新闻治理。为此，作者提出了NAGASIL方法（负采样和状态增强生成对抗自模仿学习），通过引入负采样和状态增强两项改进，提高了辟谣者选择策略的有效性。实验表明，NAGASIL在两个社交网络上的表现优于标准的GASIL和现有假新闻缓解模型。  

**Frequency Spectrum Is More Effective for Multimodal Representation and Fusion: A Multimodal Spectrum Rumor Detector**  
这项工作聚焦于社交媒体上多模态谣言检测的挑战，首次尝试在频率域内进行多模态谣言检测。传统方法注重时空位置的单模态特征混合或跨模态线索的融合，但存在单模态表示区分性弱和融合过程复杂的不足。该研究提出了一个新颖的频谱表示与融合网络（FSRU），通过傅里叶变换将空间特征转化为频谱特征，生成高度区分性强的多模态表示和融合特征。FSRU包含三个关键机制：傅里叶变换的频域转换、单模态频谱压缩以及跨模态频谱协同选择模块。实验结果表明，FSRU在多模态谣言检测任务中表现优异。  

**OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples**  
这项工作针对当前大语言模型（LLMs）生成文本难以检测的问题，提出了一个名为OUTFOX的框架，旨在提高检测器对LLM生成文本的鲁棒性。现有检测器在面对简单的改写攻击时性能会显著下降，且未考虑恶意用户可能利用检测结果规避检测的情况。OUTFOX框架通过让检测器和攻击者互相适应对方的输出进行改进：攻击者基于检测器的预测结果进行上下文学习，生成更难检测的对抗性文本；检测器则利用这些对抗性文本作为训练样本，学习应对强攻击者的检测能力。实验表明，该方法在学生作文领域中对攻击者生成文本的检测性能提升高达41.3 F1分数，并在非攻击文本检测中实现了96.9 F1分数的性能，超越现有方法。此外，提出的攻击方法显著降低了检测器性能（最高降低57.0 F1分数），优于基线改写方法。  

**Complex Claim Verification with Evidence Retrieved in the Wild**    
这项工作提出了一个现实场景下的自动事实核查流水线，针对从网络中检索原始证据以支持或反驳复杂声明的挑战。与以往假设不同，该研究限制证据检索器仅能访问在声明发布之前可用的文档，以模拟真实情况下对新兴声明的核查流程。整个流水线包括五个组件：声明分解、原始文档检索、精细证据检索、声明聚焦的摘要生成以及真实性判断。
在CLAIMDECOMP数据集上进行的实验表明，该流水线生成的聚合证据能有效提升真实性判断的准确性。人类评估进一步验证了系统生成的证据摘要具有较高的可靠性（不会虚构信息）和相关性，可回答声明的关键问题，显示其在辅助事实核查工作中具备实用性，即使未完全覆盖所有证据。  

**Faking Fake News for Real Fake News Detection: Propaganda-loaded Training Data Generation**  
这项工作针对现有假新闻检测模型在识别人类撰写的虚假信息时表现不佳的问题，提出了一种新框架，用于生成包含人类创作风格和宣传策略的训练样本。由于机器生成的假新闻与人类创作的假新闻在风格和意图上差异显著，这种差异限制了检测模型的迁移能力。
具体而言，作者利用自然语言推理（NLI）指导的自我批评序列训练，确保生成文章的真实性，同时融入诸如“权威诉求”和“情感化语言”等宣传技巧。基于此方法，构建了一个新的训练数据集PROPANEWS，包含2,256条样本，并公开供未来研究使用。  

**COSMOS: Catching Out-of-Context Misinformation with Self-Supervised Learning**  
这项工作聚焦于检测“原图配假文”——即未被篡改的图片被置于虚假的上下文中的情况，这是社交媒体上误导观众的常见方式。为支持事实核查人员，作者提出了一种自动检测图文“断章取义”的新方法。
核心思想是利用图片与文本的语义对齐（grounding），从而识别仅凭语言无法判断的上下文错误。该方法采用自监督学习策略，在训练时只需要一组带有字幕的图片，不需要显式监督。模型学会选择性地将图片中的对象与文本声明进行对齐。在测试时，检查两段字幕是否对应图片中的同一对象，但语义上却存在差异，以此准确预测图片与文本的上下文不符情况。  

**InfoSurgeon: Cross-Media Fine-grained Information Consistency Checking for Fake News Detection**  
这项研究提出了一种新颖的假新闻检测基准，基于知识单元（knowledge element）层面进行检测，同时设计了一种跨媒体一致性检查方法，以识别导致新闻文章误导性的细粒度知识单元。
为解决训练数据不足的问题，作者提出了一种新的数据合成方法，通过操纵知识图谱中的知识单元生成带有特定且难以察觉的不一致性的噪声训练数据。  

**Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News**  
将传统仅限于文本和元数据（如标题、作者）的假新闻检测，扩展到更复杂的情景，即包括图像和标题的机器生成新闻。提出了一种基于检测视觉和语义不一致性的有效方法，能够作为应对机器生成假新闻的第一道防线。  

**Document-level Claim Extraction and Decontextualisation for Fact-Checking**  
这项研究提出了一种面向事实核查的文档级声明提取方法，旨在从包含多个声明的文档中提取值得核查的声明，并使其脱离上下文独立理解。与现有的主要集中在句子级别提取声明的方法不同，本方法将声明提取视为抽取式摘要任务，首先识别文档中的核心句子，然后通过句子去上下文化的方式重写这些句子，确保它们能够独立于原始上下文理解。  

**Unveiling Opinion Evolution via Prompting and Diffusion for Short Video Fake News Detection**  
这项研究聚焦于短视频假新闻检测，提出了一种新的方法来挖掘和融合视频中显性与隐性意见的演变。当前的检测方法通常将各个模态的特征汇聚成多模态特征，但往往忽视了意见的隐性表达和跨模态意见演化的动态过程。
具体而言，作者设计了一个提示模板，从视频的文本部分挖掘关于新闻可信度的隐性意见。同时，采用了一种扩散模型，促进不同模态之间意见的相互作用，包括通过隐性意见提示所提取的意见。  

**From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News**  
这项研究提出了一个基于大语言模型（LLM）的假新闻传播仿真框架（FPS），旨在详细研究假新闻的传播趋势和控制方法。与传统的假新闻建模方法不同，该方法不仅仅预测不同群体的流行趋势或意见变化的数值化表现，而是通过模拟个体的行为和反思机制，深入探讨了人类思维模式下的意见动态。
具体来说，FPS框架中的每个代理代表一个具有独特个性的个体，具备短期和长期记忆，并能够进行反思性思考。每天，个体会随机交换意见、反思自己的想法，并更新观点。通过这种方式，模拟能够揭示与话题相关性和个体特征相关的假新闻传播模式，这些模式与现实世界的观察一致。  

**Heterogeneous Subgraph Transformer for Fake News Detection**  
本文提出了一个异质子图变换器用于假新闻检测，通过构建包含新闻话题、实体和内容关系的异质图，利用随机游走提取异质子图，并通过子图变换器进行真实性判断，取得了显著的性能提升。  

**Semantic Evolvement Enhanced Graph Autoencoder for Rumor Detection**  
这篇论文提出了一个新颖的语义演化增强图自编码器（GARD）模型，用于谣言检测。与传统方法不同，GARD模型不仅考虑了事件的传播结构信息，还捕捉了事件传播过程中语义演化的信息。通过特定的图自编码器和重建策略，GARD能够识别局部的语义变化和全局的语义演化，从而更全面地理解事件的传播过程。该模型能够更早地捕捉到谣言的语义演变，且通过引入统一性正则化器，提升了对谣言与非谣言模式的区分能力，增强了检测的准确性和鲁棒性。  

**T3RD: Test-Time Training for Rumor Detection on Social Media**  
这篇论文提出了一种新颖的Test-Time Training for Rumor Detection (T³RD) 方法，旨在提升在低资源数据集上的谣言检测性能，尤其是面对突发事件或不同语言的谣言时。传统的谣言检测模型通常依赖于大量的训练数据，并在熟悉领域表现较好，但在低资源条件下效果显著下降。为了解决这个问题，T³RD引入了自监督学习（SSL）作为辅助任务，通过全局对比学习和局部对比学习来挖掘测试样本的内在特征。全局对比学习聚焦于获取不变的图表示，局部对比学习则专注于获得不变的节点表示。此外，为了避免测试时分布偏差，模型还引入了特征对齐约束，以平衡训练集和测试样本之间的知识迁移。实验结果表明，T³RD在两个跨领域数据集上的表现达到了最新的最优水平。  

**Dual Graph Networks with Synthetic Oversampling for Imbalanced Rumor Detection on Social Media**  
这篇论文提出了一种名为Dual Graph Networks with Synthetic Oversampling (SynDGN) 的新方法，用于解决谣言检测中的类别不平衡问题。传统的谣言检测方法在面对真实信息比谣言更多的情况下，容易出现预测偏差，因为它们没有针对谣言传播的特定上下文进行适配。SynDGN通过利用双重图结构，分别整合了社交媒体上下文和用户特征来进行预测。为了缓解类别不平衡，SynDGN还引入了合成过采样技术，增强少数类样本的表现。实验结果表明，SynDGN在两个知名数据集上都优于现有的最先进模型，且不论数据是否平衡，均能保持较好的表现。  

**CMA-R:Causal Mediation Analysis for Explaining Rumour Detection**  
这篇论文提出了一种名为CMA-R (Causal Mediation Analysis for Rumor Detection) 的方法，旨在解释神经网络模型在Twitter谣言检测中的决策过程。通过对输入和网络层进行干预，CMA-R揭示了推文和单词对模型输出的因果影响。研究发现，CMA-R能够识别出关键推文，这些推文对模型预测有显著影响，并且与人类判断的真伪判断一致。此外，CMA-R还能够高亮显示在这些关键推文中的因果影响单词，为黑箱模型提供了一层额外的可解释性和透明度。实验结果表明，该方法能有效提高谣言检测模型的可解释性。  

**Style-News: Incorporating Stylized News Generation and Adversarial Verification for Neural Fake News Detection**  
这篇论文提出了一个新的验证框架Style-News，旨在检测由神经网络生成的假新闻内容。为了应对恶意社交媒体上虚假信息的传播，论文采用了出版商元数据（如文本类型、政治立场和可信度）来帮助识别新闻的风格，并利用这一风格来进行假新闻检测。该框架引入了一种风格感知的神经新闻生成器，作为对抗者，生成符合特定出版商风格的新闻内容，同时训练风格和来源判别器来识别新闻的发布者以及新闻是由人类写作还是由机器生成的。  

**Reinforced Adaptive Knowledge Learning for Multimodal Fake News Detection**  
这篇论文提出了一个名为AKA-Fake的多模态假新闻检测模型，旨在解决现有方法在处理多模态假新闻时面临的一些挑战。传统的假新闻检测方法主要关注捕捉多模态内容中的语言和视觉语义，但对于复杂且精心编造的假新闻，效果不佳。虽然一些研究尝试通过引入外部知识来增强检测能力，但现有方法往往直接使用静态实体嵌入来整合所有知识上下文，可能会引入噪声和与内容无关的知识，且难以建模多模态语义与知识实体之间的复杂关联。
AKA-Fake模型通过引入强化学习框架，针对每条新闻学习一个紧凑的知识子图，包含知识图谱中最具信息量的实体及其上下文邻居，从而有效恢复最重要的知识事实。此外，论文还提出了一个新颖的异构图学习模块，通过拓扑优化和模态注意力池化来捕捉可靠的跨模态关联。  

**Unveiling Implicit Deceptive Patterns in Multi-Modal Fake News via Neuro-Symbolic Reasoning**  
这篇文章提出了一种名为 NSLM（Neuro-Symbolic Latent Model）的新型模型，旨在解决多模态假新闻检测中的可解释性问题。与现有方法不同，NSLM不仅能够准确判断新闻的真实性，还能揭示潜在的欺骗模式，如图像操控、跨模态不一致和图像重用等。模型通过使用可学习的潜在变量和基于符号逻辑规则的弱监督学习来捕捉这些欺骗模式。此外，NSLM引入了伪孪生网络来有效区分这些模式，实验结果显示，该方法在假新闻检测中表现优越，并提供了有价值的欺骗模式解释。  

**Propagation Tree Is Not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection**  
这篇论文提出了一种名为 RAGCL（Rumor Adaptive Graph Contrastive Learning）的新方法，用于社交媒体上的谣言检测。传统的图神经网络模型假设谣言传播树（RPT）具有深层结构，但实际数据分析显示，RPT通常呈现广泛结构，节点多为浅层一层回复。为此，RAGCL通过基于节点中心性的自适应视图增强方法，聚焦于学习密集的子结构。该方法引入了三条原则来增强RPT：1）排除根节点；2）保留深层回复节点；3）在深层部分保留较低层次节点。通过节点丢弃、属性遮蔽和边缘丢弃等技术生成不同的视图，图对比学习目标则用于学习稳健的谣言表示。实验结果表明，RAGCL在四个基准数据集上超越了现有的最先进方法。该方法为谣言检测提供了一种有效的图对比学习方案，并揭示了RPT的广结构特点，具有潜在的跨领域应用价值。  

**GAMC: An Unsupervised Method for Fake News Detection using Graph Autoencoder with Masking**  
这篇论文提出了 GAMC（Graph Autoencoder with Masking and Contrastive learning）模型，旨在解决社交媒体上假新闻检测中的数据标注不足问题。传统的深度学习方法（如CNN、RNN和BERT）主要聚焦内容，忽视了新闻传播过程中的社交上下文。图神经网络技术已将社交上下文纳入考虑，但依赖大规模标注数据集仍然是其局限性。为此，GAMC通过自监督学习技术，不需要标注数据集，利用新闻传播的内容和上下文作为自监督信号。该方法通过增强新闻传播图、使用图编码器进行编码，并采用图解码器进行重建。文章提出了一种复合损失函数，包括重建误差和对比损失。通过在真实数据集上的实验验证了该方法的有效性，并展示了其在假新闻检测中的潜力。  

**Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**  
这篇论文研究了大型语言模型（LLMs）在假新闻检测中的潜力。通过实证研究，作者发现虽然GPT-3.5等先进的LLM能够提供多角度的理性分析来识别假新闻，但它们在假新闻检测任务中仍不如经过微调的小型语言模型（SLMs），如BERT。这一差距的原因在于LLM难以有效选择和整合理性分析来做出最终判断。基于此，论文提出，当前的LLM不能完全替代微调的SLM，但它可以作为SLM的辅助工具，通过提供多角度的理性指导来提升检测性能。为此，作者设计了一种适应性理性指导网络（ARG），使SLM能够从LLM的理性分析中选择性地获取信息。此外，论文还提出了ARG的无理性版本（ARGD），通过蒸馏技术在无需查询LLM的情况下进行假新闻检测。实验结果表明，ARG和ARGD在多个数据集上超过了基于SLM、LLM和它们组合的基准方法。  

**Interpretable Multimodal Misinformation Detection with Logic Reasoning**  
这篇论文提出了一种基于逻辑的神经网络模型，用于多模态虚假信息检测，旨在提升系统的可解释性和实用性。现有的多模态检测方法虽然在性能上取得了不错的结果，但缺乏可解释性，限制了这些系统的可靠性和实际应用。该模型受到神经符号AI的启发，结合了神经网络的学习能力和符号学习的可解释性。模型通过使用神经表示参数化符号逻辑元素，自动生成并评估有意义的逻辑子句，表达检测过程中的推理过程。此外，模型引入了五个元谓词，这些元谓词可以通过不同的关联进行实例化，从而增强框架在多种虚假信息来源下的泛化能力。  

**Zoom Out and Observe: News Environment Perception for Fake News Detection**  
该论文提出了一种新的假新闻检测框架——新闻环境感知框架（NEP），旨在通过观察新闻环境来捕捉有助于假新闻识别的外部信号。现有的假新闻检测方法通常聚焦于新闻内容的语言模式或验证其与知识源的匹配，忽略了假新闻的外部传播环境。NEP通过从近期主流新闻中构建宏观和微观的新闻环境，设计了两个模块：一个是基于受欢迎度的模块，另一个是基于新颖度的模块，用于捕捉与新闻传播相关的重要信号，进而辅助假新闻预测。实验结果表明，NEP框架显著提高了基本假新闻检测模型的性能。  

**Learn over Past, Evolve for Future:Forecasting Temporal Trends for Fake News Detection**  
该论文提出了一个新的假新闻检测框架——FTT（Forecasting Temporal Trends），旨在解决新闻数据快速演变所引起的时序变化问题。现有的假新闻检测方法通常在历史数据上训练并在未来数据上测试，这会导致性能显著下降。FTT框架通过预测新闻数据的时间分布模式，帮助模型选择合适的训练实例，以便更好地适应未来的数据分布。实验结果表明，FTT框架在处理时序分割数据集时具有优越的性能。  

**MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning**  
该论文提出了一种基于元学习的方法MetaAdapt，解决了在目标领域数据稀缺的情况下进行少样本误信息检测的问题。MetaAdapt通过利用有限的目标领域数据指导源领域到目标领域的知识转移，适应目标领域的变化。具体来说，MetaAdapt通过训练初始模型，计算源任务与元任务的相似度，并基于这些相似度调整梯度，以便更好地适应目标任务。实验结果表明，MetaAdapt在多个实际数据集上表现优越，能够有效地进行领域自适应的少样本误信息检测。  
