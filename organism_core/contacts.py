"""Shared contact semantics after MuJoCo constraint construction.

A constraint-active contact can have positive surface distance (margin).
An inactive gap contact is detected, but must not count as stance or support.
Active here means included in the solver, not measured force above a threshold.
"""


def is_active_contact(contact):
    """Requires mj_fwdPosition/mj_forward or a completed dynamics step.

mj_collision alone does not instantiate constraint addresses. No force is
reconstructed from poses by this predicate; it only classifies solver membership.
"""
    return int(contact.efc_address) >= 0 and int(contact.exclude) == 0
