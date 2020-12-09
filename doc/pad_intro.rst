.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:14:11 +01

.. _bob.pad.base.pad_intro:


=============================================
Introduction to presentation attack detection
=============================================

.. todo::

   Introduce PAD:

      - What it is
      - Why
      - How

   Look at bob.bio.base

Presentation Attack Detection, or PAD, is a branch of biometrics aiming at detecting an attempt to dupe a biometric recognition system by modifying the sample presented to the sensor.
The goal of PAD is to develop countermeasures to presentation attacks that are able to detect wether a biometric sample is a `bonafide` sample, or a presentation attack.

For an introduction to biometrics, take a look at the :ref:`documentation of bob.bio.base <bob.bio.base.biometrics_intro>`.

The following introduction to PAD is inspired by chapters 2.4 and 2.6 of [mohammadi2020trustworthy].

Presentation attack
===================

Biometric recognition systems contain different points of attack. Attacks on certain points are either called direct or indirect attacks.
An indirect attack would consist of modifying data after the capture, in any of the steps between the capture and the decision stages. To prevent such attacks is relevant of classical cyber security, hardware and data protection.
Presentation attacks (PA), on the other hand, are the only direct attacks that can be performed on a biometric system, and countering those attacks is relevant to biometrics.

For a face recognition system, for example, one of the possible presentation attack would be to wear a mask resembling another individual so that the system identifies the attacker as that other person.


Presentation attack detection
=============================

A PAD system works much like a biometric recognition system, but with the ability to identify and reject a sample if it is detected as an attack.
This means that multiple cases are possible and should be detected by a biometric system with PAD:

   - A Registered subject presents itself, the captured sample is called **Genuine** sample, and should be accepted by the system (positive),
   - An Attacker presents itself without trying to pass for another subject, the sample is categorized as **zero effort impostor** sample, and should be rejected by the system (negative),
   - And the special case in PAD versus "standard" biometric systems: an Attacker uses a `presentation attack instrument` (`PAI`) to pass as a genuine subject. This is a **PA** sample, and should be rejected (negative).

The term 'bona fide' is used for biometric samples presented without intention to change their identity (Genuine samples and zero effort impostor samples).


Typical implementations of PAD
------------------------------

PAD for face recognition is the most advanced in this field, face PAD systems can be categorized in several ways:

   - **Frame-based vs Video-based**: Some PAD systems classify a sample based on one image, searching for inconsistencies of resolution or lighting, and others base themselves on temporal cues like small movements or blinking.
   - **Type of light**: Some PAD systems work on visible light, using samples captured by a standard camera. A more advanced system would require a specific sensor to capture, for example, infrared light.
   - **User interaction**: Another way of asserting the authenticity of a sample is to request the presented user to respond to a challenge, like smiling or blinking at a specific moment.

PAD system using a frame-based approach on visible light with no user interaction are the least robust but are more developed, as they can be easily integrated with existing biometric systems.



References
==========

.. [mohammadi2020trustworthy]       * Mohammadi Amir **Trustworthy Face Recognition: Improving Generalization of Deep Face Presentation Attack Detection**, 2020, EPFL

.. include:: links.rst
