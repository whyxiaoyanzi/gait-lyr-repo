# Sample custom midpoint gait selection for gen_kinegraphs.sh
# Make a copy locally to change accordingly

function override(midpoint) {
  print $1, $2, $3, $4, $5, midpoint*2
  next
}
/04-11-2022_11-13-49.+Left/ { override(2) }
/04-11-2022_11-13-49.+Right/ { override(2) }
{ print $0 }
