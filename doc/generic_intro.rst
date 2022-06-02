.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:14:11 +01

.. _bob.pad.base.intro:


=============================================
Introduction to presentation attack detection
=============================================

Presentation Attack Detection, or PAD, is a branch of biometrics aiming at
detecting an attempt to dupe a biometric recognition system by modifying the
sample presented to the sensor. The goal of PAD is to develop countermeasures to
presentation attacks that can detect whether a biometric sample is a *bonafide*
sample or a presentation attack.

For an introduction to biometrics, take a look at the :ref:`documentation of
bob.bio.base <bob.bio.base.biometrics_intro>`.

The following introduction to PAD is inspired by chapters 2.4 and 2.6 of
[mohammadi2020trustworthy]_, and from [marcel2014handbook]_.


Presentation attack
===================

Biometric recognition systems contain different points of attack. Attacks on
certain points are either called direct or indirect attacks. An indirect attack
would consist of modifying data after the capture, in any of the steps between
the capture and the decision stages. To prevent such attacks is relevant to
classical cybersecurity, hardware protection, and data protection. Presentation
Attacks (PA), on the other hand, are the only direct attacks that can be
performed on a biometric system, and countering those attacks is relevant to
biometrics.

For a face recognition system, for example, one of the possible presentation
attacks would be to wear a mask resembling another individual so that the system
identifies the attacker as that other person.

New PAI (Presentation Attack Instrument) can be developed to counteract the
countermeasures put in place in the first place, so the field is in constant
evolution, to adapt to new threats and try to anticipate them.


Presentation attack detection
=============================

A PAD system works much like a biometric recognition system, but with the
expected ability to identify and reject a sample if it is detected as an attack.
This means that multiple cases are possible and should be detected by a
biometric system with PAD:

   - A Registered subject presents itself, the captured sample is called
     **Genuine** sample, and should be accepted by the system (positive),
   - An Attacker presents itself without trying to pass for another subject, the
     sample is categorized as **ZEI** (**Zero Effort Impostor**) sample, and
     should be rejected by the system (negative),
   - And the special case in PAD versus "standard" biometric systems: an
     Attacker uses a `Presentation Attack Instrument` (`PAI`) to pass as a
     genuine subject. This is a **PA** (**Presentation Attack**) sample, and
     should be rejected (negative).

The term *bonafide* is used for biometric samples presented without the
intention of changing their identity (Genuine samples and ZEI samples).

.. figure:: img/pad-classes.png
   :figwidth: 75%
   :align: center
   :alt: Four different samples organized to display the different classes of PAD.

   Categorization of samples in terms of biometric recognition and PAD systems.
   A PAD system makes the distinction between the left samples (`bonafide`,
   positives) and the right samples (`presentation attack`, negatives). In a
   biometric recognition system, genuine samples are the positives, and both
   types of impostors are the negatives.


Typical implementations of PAD
------------------------------

PAD for face recognition is the most advanced in this field, face PAD systems
can be categorized in several ways:

   - **Frame-based vs Video-based**: Some PAD systems classify a sample based on
     one image, searching for inconsistencies of resolution or lighting, and
     others base themselves on temporal cues like small movements or blinking.
   - **Type of light**: Some PAD systems work on visible light, using samples
     captured by a standard camera. A more advanced system would require a
     specific sensor to capture, for example, infrared light.
   - **User interaction**: Another way of asserting the authenticity of a sample
     is to request the presented user to respond to a challenge, like smiling or
     blinking at a specific moment.

PAD systems using a frame-based approach on visible light with no user
interaction are the least robust but are more developed, as they can be easily
integrated with existing biometric systems.


Evaluation of PAD systems
=========================

To evaluate a biometric system with PAD, a set of samples is fed to the system.
Each sample is scored, and a post-processing step is used to analyze those
scores.


Licit scenario
--------------

When no PA samples are in the input set (only Genuine and ZEI samples), the
situation is the same as a simple biometric experiment and is called a `licit`
scenario. See :ref:`biometric introduction <bob.bio.base.biometrics_intro>`.


Spoof scenario
--------------

If no ZEI samples are present in the set (only Genuine and PA samples), the
evaluation of a PAD system is seen as a two class problem, and the same
metrics as in a biometric evaluation can be used to assess its performance,
where:

   - the False Positive Rate is called IAPMR (Impostor Attack Presentation Match
     Rate),
   - the False Negative Rate is called FNMR (False Non-Match Rate),

The ROC and DET can be plotted to represent the performance of the system over a
range of operation points.

This two-class case is referred to as the `spoof` scenario.

.. note::

    To calculate IAPMR and FNMR, you must use the ``bob.bio`` packages to
    generate the relevant score files. See
    :ref:`bob.bio.base.vulnerability_analysis` for more information.


PAD evaluation
--------------

When a mix of Zero Effort Impostor and PA are present in the input set, two
possibilities arise.

The bonafide (Genuine and ZEI) samples are treated as `positives` and PA samples
are considered `negatives` (This will show the ability of the system to detect
PA). The problem becomes binary, allowing the use of similar metrics as before,
albeit with different denominations:

   - the False Positive Rate is named APCER (Attack Presentation Classification
     Error Rate),
   - the False Negative Rate is named BPCER (Bonafide Presentation
     Classification Error Rate),
   - the Half Total Error Rate is named ACER (Average Classification Error
     Rate).


The ZEI and PA samples can also be considered two separate negative classes,
leading to a ternary classification with one positive class (genuine samples)
and two distinct negative classes: ZEI and PA. The EPS (Expected Performance and
Spoofability) framework was introduced to assess the reliability of a biometric
system with PAD by defining two parameters determining how much importance is
given to each class of samples:

   - ω represents the importance of the PA scores against the ZEI scores.
   - β represents the importance of the negative classes (PA and ZEI scores)
     relative to the positive class (Genuine).

From the scores and those two parameters, the following metrics can be measured:

   - The weighted error rate for the two negative classes (IAPMR for the PA
     scores and FMR for the ZEI scores):

   :math:`\text{FAR}_\omega = \omega \cdot \text{IAPMR} + (1-\omega) \cdot \text{FMR}`

   - The weighted error rate between the previously defined
     :math:`\text{FAR}_\omega` and the :math:`\text{FNMR}` (between Genuine and
     both negatives), computed as:

   :math:`\text{WER}_{\omega.\beta} = \beta \cdot \text{FAR}_\omega + (1-\beta) \cdot \text{FNMR}`

ω and β are chosen by minimizing :math:`\text{WER}_{\omega,\beta}` on the
`development set` scores. Then by using those values, the
:math:`\text{WER}_{\omega,\beta}` is computed on the `evaluation set` scores.

.. note:: :math:`\text{HTER}_\omega` is also defined when :math:`\beta = 0.5` : :math:`\text{HTER}_\omega = {\text{FAR}_\omega + \text{FNMR} \over 2}`

The EPSC curve can be plotted to assess the performance of the system on various
ω, by fixing β. It plots the error rate :math:`\text{WER}_{\omega,\beta}`
against the weight ω. The EPSC can also be in 3D if β is not fixed, showing the
:math:`\text{WER}_{\omega,\beta}` against both weights ω and β.


References
==========

.. [mohammadi2020trustworthy]       * Mohammadi Amir **Trustworthy Face Recognition: Improving Generalization of Deep Face Presentation Attack Detection**, 2020, EPFL
.. [marcel2014handbook]             * Marcel, Sébastien and Nixon, Mark S and Li, Stan Z **Handbook of biometric anti-spoofing**, 2014, Springer

.. include:: links.rst
