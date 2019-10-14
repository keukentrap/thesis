Malware is a steadily growing problem with new samples occuring every day. (McAfee Labs Threats Report August 2019) but the complexity of the malware has also grown. Cyber-threats have extended from individual or small organized groups to progessional intelligence officers learding state-sponsored cyber-missions. 
-- More why this is important.

Classifying Malware samples with machine learning is a relatively new field.

Analysis of Malware can be divided into two categories.
The first is dynamic analysis and the latter is static analysis.
## Dynamic Analysis
Dynamic analysis means running the executable and analysis its behaviour inside a sandboxed environment, for example an emulator.
Most of the times, this gives a clearer picture about the malware and it's identity, but there are downsides. Running each binary your want to classify dynamically is very resoursce intensive, because each time the needs to be a fresh emulator ready to analyse. You also have to run the binary, and without knowing what is will do this may impose risks. Malware could also detect the sandboxed environment and not show their behaviour. (Reffetseder, Kreugel, and Kirda 2007) Dynamic analysis implies you can run the binary, this is not always the case.

Previous work has made multiple attempts to classify malware with a dynamic approach. By analysing the syscall sequence (Kolosnjaji et al. 2016) or by analysing virus scan reports on the behaviour of the malware. (Boot, Poll, Serban 2019)



## Static Analysis
Static Analysis means that you only look at the binary of the malware sample. You can identify what syscall will be executed, for what OS it is written for and in what language the program was written. However, static analysis is susceptible to obfustication of the binary. A packer can compress and encrypt a binary to conceal its content. One may think that only malware packs its binaries, but this is not the case(Guo,Ferrie, and Chiueh 2008)


