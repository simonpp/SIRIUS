---
layout: post
title: Template for Fortran-C interface
---
C side:
{% highlight C %}
void sirius_add_atom(char*   label__,
                     double* position__,
                     double* vector_field__)
{
    if (vector_field__ != NULL) {
        do_something();
    } else {
        do_something_else();
    }
}
{% endhighlight %}

Fortran side:

{% highlight fortran %}
subroutine sirius_add_atom(label, pos, vfield)
    implicit none
    character,         target, dimension(*), intent(in) :: label
    real(8),           target,               intent(in) :: pos
    real(8), optional, target,               intent(in) :: vfield
    type(C_PTR) label_ptr, pos_ptr, vfield_ptr

    interface
        subroutine sirius_add_atom_aux(label, pos, vfield)&
            &bind(C, name="sirius_add_atom")
            use, intrinsic :: ISO_C_BINDING
            type(C_PTR), value, intent(in) :: label
            type(C_PTR), value, intent(in) :: pos
            type(C_PTR), value, intent(in) :: vfield
        end subroutine
    end interface

    label_ptr = C_LOC(label(1))
    pos_ptr = C_LOC(pos)
    vfield_ptr = C_NULL_PTR
    if (present(vfield)) vfield_ptr = C_LOC(vfield)

    call sirius_add_atom_aux(label_ptr, pos_ptr, vfield_ptr)

end subroutine
{% endhighlight %}
