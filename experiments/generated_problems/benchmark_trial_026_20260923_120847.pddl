(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    cube - obj
    blue_can - obj
    pot - location
    bin - location
  )
  (:init
    (on-table cube)
    (on-table blue_can)
  )
  (:goal (and
    (on cube pot)
    (on blue_can bin)
  ))
)
