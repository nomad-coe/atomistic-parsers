gmx grompp -f water.298.mdp -c water.confout.gro -p water.top -o water.tpr

gmx mdrun -s water.tpr -o water.trr
