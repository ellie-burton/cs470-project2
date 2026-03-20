Assignment



The assignment is to animate 3 algorithms.

#1 The Hungarian algorithm for min/max assignment.

#2 Gale Shapley algorithm for stable (marriage) assignment

#3 Dinic's Algorithm for max flow.




Smaller teams may seek some relief. All team members are expected to participate. And it is hoped that the teams may find some reusable code particularly in the animation/graphics stages. In the end we are looking either for a single code base or 4 code bases with one being a shared library.



Animation algorithms can be tricky. The general process is to step though loops animating the results of each pass. Most graphics engines are interrupt driven so the function composing the animation has to end before each drawing will appear.



A common mistake is to build something like:

void paint(graphics g) {

     while (1) {

           doStuff()

           g.drawStuffdone()

           sleep(1);

     }

}

and then nothing happens until the function ends (you only see the final drawing). This happens because paint usually works in memory buffer and only when the paint function ends is the buffer copied to output graphics using a bit block transfer (bitblt) function.

The correct structure is more like:



void paint(graphics g) {

    g.drawStuffdone();

    async.after(1) {doStuff()}//call this later!

}

void doStuff(){

   ...work done

   repaint(); //paint isn't called immediately but the requested as soon as idle

}



or

void paint(graphics g) {

    g.drawStuffdone();

}

void doStuff(){

   ...work is done here

}

void main(){

     Timer.repeat(1, {doStuff(); repaint();} ):

}



I'll try to post some simple examples.

Easy enough in Java Swing, Python+TKinter with no additional requirements (these are baked into the language). Javascript/html too.



In C/C++ in hear good things about raylib but never used it. If it's simple enough to describe the download we'll allow it or similar.



Your solution must be portable. If the grader cannot run it you have a problem, and I do not know the grader's workstation setup. Nor do I plan to ask.



The animations should focus on the algorithms. Not simply the results. I want the animation to illustrate how the algorithms work. This is more important than illustrating the outcome of the matching.