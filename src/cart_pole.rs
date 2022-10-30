extern crate find_folder;

use crate::{ActionType, GifRender, GymEnv};
use std::collections::HashMap;
// use plotters::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;

/**
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along
    a frictionless track. The pendulum starts upright, and the goal is to
    prevent it from falling over by increasing and reducing the cart's
    velocity.

Source:
    https://github.com/openai/gym
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson

Observation:
    Type: Vec<f64>
    Index   Observation             Min                     Max
    0       Cart Position           -4.8                    4.8
    1       Cart Velocity           -Inf                    Inf
    2       Pole Angle              -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity   -Inf                    Inf

Actions:
    Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    Note: The amount the velocity that is reduced or increased is not
    fixed; it depends on the angle the pole is pointing. This is because
    the center of gravity of the pole increases the amount of energy needed
    to move the cart underneath it

Reward:
    Reward is 1 for every step taken, including the termination step

Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]

 Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of
    the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to
    195.0 over 100 consecutive trials.
**/
#[derive(Debug)]
pub struct CartPoleEnv {
    gravity: f32,
    #[allow(dead_code)]
    mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    length: f32, // actually half the pole's length
    pole_mass_length: f32,
    force_mag: f32,
    tau: f32, // seconds between state updates
    kinematics_integrator: KinematicsIntegrator,
    // Angle at which to fail the episode
    theta_threshold_radians: f32,
    x_threshold: f32,
    rng: StdRng,
    state: [f32; 4],
    episodic_return: f32,
    episodic_length: u32,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum KinematicsIntegrator {
    Euler,
    SemiImplicitEuler,
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let length = 0.5;
        Self {
            gravity: 9.8,
            mass_cart,
            mass_pole,
            total_mass: mass_cart + mass_pole,
            length,
            pole_mass_length: mass_pole * length,
            force_mag: 10.0,
            tau: 0.02,
            kinematics_integrator: KinematicsIntegrator::Euler,
            theta_threshold_radians: 12.0 * 2.0 * std::f32::consts::PI / 360.0,
            x_threshold: 2.4,
            rng: StdRng::from_entropy(),
            state: [0.0; 4],
            episodic_return: 0.0,
            episodic_length: 0,
        }
    }
}

impl GymEnv for CartPoleEnv {
    fn step(
        &mut self,
        action: ActionType,
    ) -> (Vec<f32>, f32, bool, Option<HashMap<String, String>>) {
        let action = match action {
            ActionType::Discrete(v) => v,
            // ActionType::Continuous(_) => panic!("wrong action type provided"),
        };

        let mut x = self.state[0];
        let mut x_dot = self.state[1];
        let mut theta = self.state[2];
        let mut theta_dot = self.state[3];

        let force = if action == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp =
            (force + self.pole_mass_length * theta_dot.powi(2) * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta.powi(2) / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;

        match self.kinematics_integrator {
            KinematicsIntegrator::Euler => {
                x += self.tau * x_dot;
                x_dot += self.tau * x_acc;
                theta += self.tau * theta_dot;
                theta_dot += self.tau * theta_acc;
            }
            KinematicsIntegrator::SemiImplicitEuler => {
                x_dot += self.tau * x_acc;
                x += self.tau * x_dot;
                theta_dot += self.tau * theta_acc;
                theta += self.tau * theta_dot;
            }
        }
        self.state = [x, x_dot, theta, theta_dot];

        let done: bool = x < -self.x_threshold
            || x > self.x_threshold
            || theta < -self.theta_threshold_radians
            || theta > self.theta_threshold_radians;

        let reward: f32 = match done {
            true => 0.,
            false => 1.,
        };

        self.episodic_return += reward;
        self.episodic_length += 1;

        return if done {
            let info = Some(HashMap::from([
                (
                    "episodic_return".parse().unwrap(),
                    self.episodic_return.to_string(),
                ),
                (
                    "episodic_length".parse().unwrap(),
                    self.episodic_length.to_string(),
                ),
            ]));
            (self.reset(), reward, done, info)
        } else {
            (self.state.to_vec(), reward, done, None)
        };
    }

    fn reset(&mut self) -> Vec<f32> {
        let d = Uniform::new(-0.05, 0.05);
        self.state = [
            self.rng.sample(d),
            self.rng.sample(d),
            self.rng.sample(d),
            self.rng.sample(d),
        ];
        self.episodic_return = 0.0;
        self.episodic_length = 0;

        self.state.to_vec()
    }

    #[allow(unused_variables)]
    fn render(&self, render: &mut GifRender) {
        // render.drawing_area.fill(&WHITE).unwrap();
        //
        // let mut chart = ChartBuilder::on(&render.drawing_area)
        //     .caption(format!("Cart Pole Environment"), ("sans-serif", 20))
        //     .build_cartesian_2d(-2.4..2.4, 0_f64..1_f64)
        //     .unwrap();
        //
        // // draw track
        // let track_y: f32 = 0.25;
        // chart
        //     .draw_series(LineSeries::new(
        //         vec![(-2.4, track_y), (2.4, track_y)],
        //         &BLACK,
        //     ))
        //     .unwrap();
        //
        // // draw cart
        // let cart_x = self.state[0];
        // let cart_width = 0.0833;
        // let cart_height = 0.075;
        // chart
        //     .draw_series(vec![(0.0, 0.0)].iter().map(|_| {
        //         Rectangle::new(
        //             [
        //                 ((cart_x - cart_width), track_y),
        //                 ((cart_x + cart_width), (track_y + cart_height)),
        //             ],
        //             HSLColor(0.8, 0.7, 0.1).filled(),
        //         )
        //     }))
        //     .unwrap();
        //
        // // draw pole
        // let pole_angle = self.state[2];
        // let pole_top_x = cart_x + (pole_angle).sin() * self.length;
        // let pole_top_y = cart_height + track_y + (pole_angle).cos() * self.length;
        // chart
        //     .draw_series(LineSeries::new(
        //         vec![(cart_x, track_y + cart_height), (pole_top_x, pole_top_y)],
        //         &RED,
        //     ))
        //     .unwrap();
        //
        // // draw score
        // let style = TextStyle::from(("sans-serif", 20).into_font()).color(&RED);
        // render
        //     .drawing_area
        //     .draw_text(
        //         &format!("Score: {}", self.score),
        //         &style,
        //         (
        //             scale(0.0, 1.0, 0.0, render.width as f64, 0.1) as i32,
        //             scale(0.0, 1.0, 0.0, render.height as f64, 0.9) as i32,
        //         ),
        //     )
        //     .unwrap();
        //
        // render.drawing_area.present().unwrap()
    }

    fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}
