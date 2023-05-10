# Software-Defined Network Traffic Engineering

This is our final project for (Fall 2023) CS 4450 – Introduction to Computer Networks at Cornell University. The objectives for this project was to develop and analyze algorithms for software-defined network traffic engineering using linear programming. One algorithm was made to maximize a network's total throughput and another to minimize the maximum link utilization. For this project we used the topology and demands as described the B4 and Sprint folders at this [repository](https://github.com/manyaghobadi/teavar/tree/master/code/data).

## Authors

- Richard Kim (rk625) [@richardshkimm](https://www.github.com/richardshkimm)
- Grace Ge (gg398) [@gracege678](https://www.github.com/gracege678)
- Willy Jiang (wjj26) [@wjjiang1](https://www.github.com/wjjiang1)
- Rebecca Hasser (reh289) [@rhasser](https://www.github.com/rhasser)
- Maelat Mekonen (mmm432) [@mmmekonen](https://www.github.com/mmmekonen)

## Run Locally

#### Make sure you have Gurobipy, NetworkX, and Matplotlib installed!

Clone the project

```bash
  git clone https://github.com/richardshkimm/te-project-dao.git
```

Go to the project directory

```bash
  cd te-project-dao
```

Running the code snippet below will output the results of the Gurobipy optimization in the terminal and produce an image for the flow allocation graph

```bash
  python3 main.py
```

To see the solving times for smaller vs larger topologies run the Jupyter Notebook: extra_credit.ipynb
