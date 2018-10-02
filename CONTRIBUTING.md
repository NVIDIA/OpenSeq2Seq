# Contributing

We welcome your contributions to OpenSeq2Seq!

# Issues

## Questions
If you are submitting a question - please tag your issue with ``question`` label. 

## Feature requests
Describe your feature request in detail and use ``enchancement`` tag.

## Bugs
For all other issues make sure you've included all the necessary information for us 
to reproduce your problem:
 * TensorFlow version
 * OpenSeq2Seq version (or commit id)
 * GPU models, CUDA and driver versions
 * etc.
 
# Pull requests
If you want to start working on OpenSeq2Seq a good place to start would be issues tagged with ``good first issue`` label.
You might also consider opening an ``enchancement`` issue and assigning it yourself, especially if you plan a bigger change.

* The best way to send a PR is to fork OpenSeq2Seq and send PR from your fork
* Before merging, your PR will have to go through code review by us

# Sign your work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the --signoff (or -s) option when committing your changes:

$ git commit -s -m "Add cool feature."

This will append the following to your commit message:

Signed-off-by: Your Name <your@email.com>

By doing this you certify the below:

Developer Certificate of Origin  
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.  
1 Letterman Drive  
Suite D4700  
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

# Code Style

We use [Google Python Code Style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md), so make sure to run pylint over your code (pylint config is located in the `.pylintrc` file in the root of OpenSeq2Seq) and check that the score is >= 8.0. We also recommend using [GIT Pylint Commit Hook](https://git-pylint-commit-hook.readthedocs.io/en/latest/usage.html) to automate this process.

Thanks for your contributions!
