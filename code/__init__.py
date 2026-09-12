"""
MSFD Replication Package

A modular implementation of Multi-Headed Synthetic Feature Distillation (MSFD)
for the ReScience C replication study.

Usage:
    from code.config import Config
    from code.models import MultimodalTeacher, LightweightStudent
    from code.training import train_teacher, train_baseline
    from code.distillation import train_student_messy, train_standard_kd
"""
__version__ = "1.0.0"