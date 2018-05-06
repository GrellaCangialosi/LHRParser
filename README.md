# LHRParser (Latent Heads Representation)

This is an implementation in [Kotlin](https://kotlinlang.org/) of the parser described in 
[Non-Projective Dependency Parsing via Latent Heads Representation (LHR)](https://arxiv.org/abs/1802.02116 
"see on arxiv.org"), based on the [SimpleDNN](https://github.com/kotlinnlp/SimpleDNN "SimpleDNN on GitHub") neural 
network library.

LHRParser is a neural dependency parser that implements a novel approach based on a bidirectional recurrent autoencoder 
to perform globally optimized non-projective parsing via semi-supervised learning.

![LHR Parser schema](https://dl.dropboxusercontent.com/s/5p71fzp0hveqktl/paper_MQ.gif)
The image shows the architecture, composed by two BiLSTM encoders that produce the Context Vectors and the Latent 
Heads, a Similarity decoder that finds the relations between dependents and governors and the Labeler that assigns 
a dependency relation label and a part-of-speech tag to each token.


## Examples

Try some examples of training and evaluation of LHRParser running the files in the `examples` folder.


## Citation

If you make use of this software for research purposes, we'll appreciate citing the following:

    @ARTICLE{2018arXiv180202116G, 
        author = {{Grella}, M. and {Cangialosi}, S.}, 
        title = "{Non-Projective Dependency Parsing via Latent Heads Representation (LHR)}", 
        journal = {ArXiv e-prints}, 
        archivePrefix = "arXiv", 
        eprint = {1802.02116}, 
        primaryClass = "cs.CL", 
        keywords = {Computer Science - Computation and Language}, 
        year = 2018, 
        month = feb, 
        adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180202116G}, 
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


## Contact

For questions and usage issues, please contact [matteogrella@gmail.com](mailto:matteogrella@gmail.com) or 
[sm.cangialosi@gmail.com](mailto:sm.cangialosi@gmail.com).


## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")
