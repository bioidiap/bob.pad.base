.. vim: set fileencoding=utf-8 :
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: 2020-11-27 15:14:11 +01

.. _bob.pad.base.pad_intro:


===============================================
 Introduction to presentation attack detection
===============================================

.. todo::

   Introduce PAD:

      - What it is
      - Why
      - How

   Look at bob.bio.base

Presentation Attack Detection, or PAD, is a branch of biometrics aiming at detecting an attempt to dupe a biometric recognition system by modifying the sample presented to the sensor.
The goal of PAD is to develop countermeasures to presentation attacks that are able to detect wether a biometric sample is a `bonafide` sample, or a presentation attack.
The paper

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

   - A Genuine subject presents itself, the captured sample is called **bona fide** sample,
   - An Attacker presents itself without trying to pass for another subject, the sample is categorized as **zero effort impostor** sample,
   - An Attacker uses a `presentation attack instrument` (`PAI`) to pass as a genuine subject. This is a **PA** sample.
