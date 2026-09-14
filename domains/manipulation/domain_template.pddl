(define (domain manipulation)
 (:requirements :strips :typing) 
 (:types robot obj location)
 (:predicates)

 (:action pick
   :parameters (?obj)
   :precondition ()
   :effect ()
 )

 (:action place
   :parameters (?obj ?pos)
   :precondition ()
   :effect ()
 )
)