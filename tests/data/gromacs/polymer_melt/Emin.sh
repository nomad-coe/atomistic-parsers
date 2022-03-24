init="step3_input"
mini_prefix="step4.0_minimization"
#set equi_prefix = step4.1_equilibration
#set prod_prefix = step5_production
#set prod_step   = step5

# Minimization
# In the case that there is a problem during minimization using a single precision of GROMACS, please try to use
# a double precision of GROMACS only for the minimization step.
#grompp -f ${mini_prefix}.mdp -o ${mini_prefix}.tpr -c ${init}.gro -r ${init}.gro -p topol.top -n index.ndx -maxwarn -1
mdrun -v -deffnm ${mini_prefix}
